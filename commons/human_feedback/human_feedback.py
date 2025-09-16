"""
human_feedback.py implements human feedback generation.

incoming HumanFeedbackRequests should trigger generation of hf tasks
- concurrently generate hf tasks for each miner_feedback
- store completed HumanFeedbackResponse in redis
    key: 3 x miner_response_id + validator_task_id?
    value: HumanFeedbackResponse

"""

import asyncio
import random
import traceback
from typing import List

from langfuse import observe
from loguru import logger
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
)

from commons.cache import RedisCache
from commons.config import ANSWER_MODELS
from commons.human_feedback.hf_prompts import get_hf_prompt
from commons.human_feedback.types import (
    HumanFeedbackRequest,
    HumanFeedbackResponse,
    HumanFeedbackTask,
    ImprovedCode,
)
from commons.linter import lint_and_fix_code
from commons.llm import call_llm, get_llm_api_client
from commons.synthetic import (
    CodeAnswer,
    merge_js_and_html,
)
from commons.utils import log_to_langfuse


async def generate_human_feedback(
    hf_request: HumanFeedbackRequest,
    hf_id: str,
) -> None:
    """
    Driver func to generate human feedback for a given request.
    """
    redis = RedisCache()
    try:
        logger.info(f"generating human feedback for {hf_id}")
        # concurrently generate improved outputs from the miner feedbacks
        async_tasks = [
            _generate_human_feedback(
                hf_request.base_prompt,
                hf_request.base_code,
                miner_feedback.feedback,
                miner_feedback.miner_response_id,
            )
            for miner_feedback in hf_request.miner_feedbacks
        ]
        improved_codes: List[ImprovedCode] = await asyncio.gather(*async_tasks)
        logger.info(f"hf generation for {hf_id} completed")
        improved_id_to_code_map = {
            data.miner_response_id: data.code for data in improved_codes
        }

        # Create HumanFeedbackTasks from improved code
        hf_tasks: List[HumanFeedbackTask] = []
        for miner_feedback in hf_request.miner_feedbacks:
            # find matching improved code for the miner_response_id
            corresponding_code = improved_id_to_code_map[
                miner_feedback.miner_response_id
            ]

            _hf_task = HumanFeedbackTask(
                miner_hotkey=miner_feedback.hotkey,
                miner_response_id=miner_feedback.miner_response_id,
                feedback=miner_feedback.feedback,
                model=random.choice(ANSWER_MODELS),
                generated_code=corresponding_code,
            )
            hf_tasks.append(_hf_task)

        # compile into response object
        hf_resp = HumanFeedbackResponse(
            success=True,
            hf_id=hf_id,
            base_prompt=hf_request.base_prompt,
            base_code=hf_request.base_code,
            human_feedback_tasks=hf_tasks,
        )
        # store completed HumanFeedbackResponse in redis
        await redis.store_human_feedback(hf_id=hf_id, data=hf_resp)
    except Exception as e:
        # @dev: what happens if redis fails here
        error_trace = traceback.format_exc()
        logger.error(f"Error generating human feedback: {hf_id} {e}\n{error_trace}")
        await redis.store_human_feedback(hf_id=hf_id, data=None)


@observe(as_type="generation", capture_input=True, capture_output=True)
async def _generate_human_feedback(
    base_prompt: str,
    base_code: str,
    feedback: str,
    miner_response_id: str,
    attempt_count: int = 1,
) -> ImprovedCode:
    """
    private func to generate improved code with a the base_prompt and base_code and feedback.

    """
    model = random.choice(ANSWER_MODELS)
    client = get_llm_api_client()

    messages = [
        {
            "role": "system",
            "content": get_hf_prompt(base_prompt, base_code, feedback),
        }
    ]

    kwargs = {
        "response_model": CodeAnswer,
        "model": model,
        "messages": messages,
        "max_retries": AsyncRetrying(stop=stop_after_attempt(2), reraise=True),
        "temperature": random.uniform(0, 1),
        "top_p": random.uniform(0, 0.8),
    }
    logger.info(
        f"generating improved output for miner_response_id: {miner_response_id}"
    )

    try:
        completion, raw_completion = await call_llm(client, kwargs)
        log_to_langfuse(kwargs, completion, raw_completion)

        # check for unmerged output
        # TO-DO: enable linting javascript that is already in a .html file
        for f in completion.files:
            if f.filename.endswith(".js") and len(f.content) > 1:
                logger.info(f"linting and fixing {f.filename}")  # remove
                completion = await lint_and_fix_code(
                    client, model, completion, miner_response_id
                )
                completion = merge_js_and_html(completion)
                break

        # check that improved code is different from base code
        improved_code_str = completion.model_dump_json()
        if base_code == improved_code_str:
            logger.error(
                f"improved code identical with base code for miner response: {miner_response_id} attempt: {attempt_count}"
            )
            if attempt_count < 3:
                return await _generate_human_feedback(
                    base_prompt,
                    base_code,
                    feedback,
                    miner_response_id,
                    attempt_count + 1,
                )
            else:
                logger.error(
                    f"failed to generate unique HFL task after max retries for miner_response_id: {miner_response_id}"
                )
                raise Exception("failed to generate unique HFL task after max retries")

        return ImprovedCode(code=completion, miner_response_id=miner_response_id)
    except Exception as e:
        logger.error(
            f"Error generating human feedback miner_response_id: {miner_response_id} {e}"
        )
        raise e


# async def main():
#     """
#     for testing human feedback generation. Remove in prod.
#     """

#     from commons.human_feedback.types import MinerFeedback

#     logger.info("testing human feedback ...")
#     # dummy data
#     dummy_prompt = """Create an interactive referee decision simulator that demonstrates nuanced football rules through visual feedback.

#     Features:
#     - Display a top-down football field with clearly marked penalty areas and midfield line using HTML canvas
#     - Implement a referee character that can perform 3 animations: showing yellow card, red card, and pointing for penalty kick
#     - Create clickable foul zones with different consequences:
#     * Penalty area (goalkeeper collision)
#     * Midfield (tackle challenge)
#     * Wing area (handball scenario)
#     - Include a severity slider that adjusts from "Incidental Contact" to "Reckless Challenge"
#     - Add a toggle switch for "Intentional" vs "Unintentional" fouls
#     - When inputs change, display real-time visual feedback:
#     - Field zone highlights with pulsating borders
#     - Dynamic text rulings that explain referee's decision logic
#     - Animated cards that emerge from referee's pocket with physics-based motion
#     - Contextual hand signals that match FIFA officiating standards
#     - Implement particle effects for card reveals (gold sparkles for yellow, red glow for red)
#     - Include a "Challenge Flag" button that triggers VAR-style video replay lines over the foul area

#     User Actions:
#     1. Click different field zones to simulate foul locations
#     2. Adjust severity slider to modify foul intensity
#     3. Toggle intentional/unintentional switch to change foul context

#     Note: The visualization emphasizes referee decision-making subtleties through interactive elements, mirroring real-world officiating complexities discussed during casual analysis.
#     Note:
#     - Your output should be implemented in JavaScript with HTML and CSS.
#     - Ensure that the output has both index.js and index.html files
#     """

#     dummy_code_answer = {
#         "files": [
#             {
#                 "filename": "index.js",
#                 "content": "const canvas=document.getElementById('argumentCanvas'),ctx=canvas.getContext('2d'),simplifyBtn=document.getElementById('simplifyBtn'),resetBtn=document.getElementById('resetBtn');let width,height,nodes=[],links=[],isDragging=!1,draggedNode=null,offsetX,offsetY,isSimplified=!1,animationStartTime=0,animationDuration=1e3,animating=!1,targetNodePositions=[];const backgroundColor='#F0F8FF',nodeColor='#2C3E50',initialLineColor='#B0C4DE',simplifiedLineColor='#2ECC71',buttonBgColor='#34495E',buttonTextColor='#FFFFFF',nodeRadius=25,nodeLabelColor='#FFFFFF';function setupCanvas(){width=Math.min(.9*window.innerWidth,.9*window.innerHeight),height=width,canvas.width=width,canvas.height=height,ctx.clearRect(0,0,width,height),ctx.fillStyle=backgroundColor,ctx.fillRect(0,0,width,height),generateInitialArgument(),draw()}function generateInitialArgument(){nodes=[],links=[],isSimplified=!1;const e=[\"Point A\",\"Fact 1\",\"Evidence 2\",\"Premise C\",\"Conclusion\",\"Data X\",\"Theory Y\",\"Case Z\",\"Principle P\",\"Rule Q\",\"Finding R\",\"Hypothesis S\"],t=12;for(let n=0;n<t;n++){const t=Math.random()*(.6*width)+.2*width,s=Math.random()*(.6*height)+.2*height;nodes.push({id:n,label:e[n%e.length],x:t,y:s,initialX:t,initialY:s,targetX:t,targetY:s,currentX:t,currentY:s})}for(let e=0;e<t;e++){let n=Math.floor(3*Math.random())+1;for(let s=0;s<n;s++){let n=Math.floor(Math.random()*t);(n===e||links.some(t=>t.source===e&&t.target===n||t.source===n&&t.target===e))||links.push({source:e,target:n})}}}function drawLinks(){links.forEach(e=>{const t=nodes[e.source],n=nodes[e.target];ctx.beginPath(),ctx.moveTo(t.currentX,t.currentY),ctx.lineTo(n.currentX,n.currentY),isSimplified||animating?(function(e,t){const n=(Date.now()-animationStartTime)/animationDuration,s=1+2*n,o=parseInt(initialLineColor.substring(1,3),16),i=parseInt(initialLineColor.substring(3,5),16),a=parseInt(initialLineColor.substring(5,7),16),r=parseInt(simplifiedLineColor.substring(1,3),16),l=parseInt(simplifiedLineColor.substring(3,5),16),c=parseInt(simplifiedLineColor.substring(5,7),16),d=Math.round(o+(r-o)*n),u=Math.round(i+(l-i)*n),h=Math.round(a+(c-a)*n);ctx.strokeStyle=`rgb(${d},${u},${h})`,ctx.lineWidth=s})(0,0): (ctx.strokeStyle=initialLineColor,ctx.lineWidth=1),ctx.stroke()})}function drawNodes(){nodes.forEach(e=>{ctx.beginPath(),ctx.arc(e.currentX,e.currentY,nodeRadius,0,2*Math.PI),ctx.fillStyle=nodeColor,ctx.fill(),ctx.strokeStyle='#FFFFFF',ctx.lineWidth=1,ctx.stroke(),ctx.fillStyle=nodeLabelColor,ctx.font='12px Arial',ctx.textAlign='center',ctx.textBaseline='middle',ctx.fillText(e.label,e.currentX,e.currentY)})}function draw(){ctx.clearRect(0,0,width,height),ctx.fillStyle=backgroundColor,ctx.fillRect(0,0,width,height),drawLinks(),drawNodes()}function animate(){if(animating){const e=Date.now()-animationStartTime,t=Math.min(e/animationDuration,1);nodes.forEach(e=>{e.currentX=e.x+(e.targetX-e.x)*t,e.currentY=e.y+(e.targetY-e.y)*t}),draw(),t<1?requestAnimationFrame(animate):(animating=!1,isSimplified=!0,nodes.forEach(e=>{e.x=e.targetX,e.y=e.targetY}),draw())}else draw()}canvas.addEventListener('mousedown',e=>{e.preventDefault();const t=canvas.getBoundingClientRect(),n=e.clientX-t.left,s=e.clientY-t.top;for(let e=nodes.length-1;e>=0;e--){const t=nodes[e],o=Math.sqrt(Math.pow(n-t.currentX,2)+Math.pow(s-t.currentY,2));if(o<nodeRadius){isDragging=!0,draggedNode=t,offsetX=n-t.currentX,offsetY=s-t.currentY,isSimplified=!1;break}}}),canvas.addEventListener('mousemove',e=>{if(!isDragging||!draggedNode)return;e.preventDefault();const t=canvas.getBoundingClientRect(),n=e.clientX-t.left,s=e.clientY-t.top;draggedNode.x=n-offsetX,draggedNode.y=s-offsetY,draggedNode.currentX=draggedNode.x,draggedNode.currentY=draggedNode.y,draggedNode.x=Math.max(nodeRadius,Math.min(width-nodeRadius,draggedNode.x)),draggedNode.y=Math.max(nodeRadius,Math.min(height-nodeRadius,draggedNode.y)),draggedNode.currentX=draggedNode.x,draggedNode.currentY=draggedNode.y,draw()}),canvas.addEventListener('mouseup',()=>{isDragging=!1,draggedNode=null}),canvas.addEventListener('mouseleave',()=>{isDragging=!1,draggedNode=null}),simplifyBtn.addEventListener('click',()=>{if(animating)return;const e=width/2,t=height/2,n=.35*Math.min(width,height);nodes.forEach((s,o)=>{const i=o/nodes.length*2*Math.PI;s.targetX=e+n*Math.cos(i),s.targetY=t+n*Math.sin(i),s.x=s.currentX,s.y=s.currentY}),animationStartTime=Date.now(),animating=!0,requestAnimationFrame(animate)}),resetBtn.addEventListener('click',()=>{if(animating)return;nodes.forEach(e=>{e.targetX=e.initialX,e.targetY=e.initialY,e.x=e.currentX,e.y=e.currentY}),animationStartTime=Date.now(),animating=!0,isSimplified=!1,requestAnimationFrame(animate)}),window.addEventListener('resize',setupCanvas),setupCanvas();const instructionsDiv=document.createElement('div');instructionsDiv.style.cssText='position: fixed; top: 10px; left: 50%; transform: translateX(-50%); background: rgba(0, 0, 0, 0.6); color: #FFFFFF; padding: 8px 15px; border-radius: 5px; font-family: Arial, sans-serif; font-size: 14px; text-align: center; z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.2);',instructionsDiv.innerHTML='Drag nodes to untangle. Click \"Simplify Argument\" for automatic organization.',document.body.appendChild(instructionsDiv);",
#             },
#             {
#                 "filename": "index.html",
#                 "content": '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><meta http-equiv="Feature-Policy" content=" camera \'none\'; microphone \'none\'; geolocation \'none\'; accelerometer \'none\'; gyroscope \'none\'; magnetometer \'none\'; payment \'none\'; usb \'none\';"><title>Argument Simplifier</title><style>body{margin:0;display:flex;flex-direction:column;justify-content:center;align-items:center;min-height:100vh;background-color:#F0F8FF;font-family:Arial,sans-serif;overflow:hidden}canvas{border:1px solid #B0C4DE;background-color:#F0F8FF;display:block;box-shadow:0 4px 10px rgba(0,0,0,0.1);border-radius:8px;cursor:grab}canvas:active{cursor:grabbing}.controls{position:fixed;bottom:20px;display:flex;gap:15px;z-index:10}.btn{background-color:#34495E;color:#FFFFFF;border:none;padding:12px 25px;border-radius:30px;cursor:pointer;font-size:16px;font-weight:bold;transition:background-color .3s ease,transform .1s ease;box-shadow:0 3px 8px rgba(0,0,0,0.2)}.btn:hover{background-color:#2C3E50;transform:translateY(-1px)}.btn:active{transform:translateY(0);box-shadow:0 1px 3px rgba(0,0,0,0.2)}</style></head><body><canvas id="argumentCanvas"></canvas><div class="controls"><button id="simplifyBtn" class="btn">Simplify Argument</button><button id="resetBtn" class="btn">Reset Argument</button></div><script src="index.js"></script></body></html>',
#             },
#         ]
#     }

#     dummy_q_2 = """
#      Create an interactive visualization that demonstrates the process of simplifying complex information, inspired by the clarity and structure of legal arguments.

#     Features:
#     - A central display area should present a network of interconnected nodes, representing a "convoluted argument." The nodes and their connecting lines must be initially tangled and overlapping, creating a visually complex and messy appearance.
#     - Each node should be a distinct circular element, with a simple, short label (e.g., "Point A", "Fact 1", "Conclusion"). There should be at least 10-15 nodes.
#     - Lines connecting the nodes represent logical links. These lines should initially be thin and light gray, emphasizing their entanglement and lack of clarity.
#     - Users can click and drag any node to reposition it. As a node is dragged, its connected lines should dynamically update, stretching and moving with it, allowing the user to manually untangle the web.
#     - A prominent button labeled "Simplify Argument" should be present. When clicked, this button triggers an animated transition where the nodes automatically re-arrange themselves into a more organized, less tangled, and visually clearer structure. During this transition, the connecting lines should become thicker and change to a vibrant color to signify clarity and a simplified logical flow.
#     - A button labeled "Reset Argument" should be available. When clicked, it returns the network of nodes and lines to its initial convoluted and tangled state.
#     - The color scheme should be professional and clear. The background should be a very light blue or off-white (e.g., `#F0F8FF` or `#F5F5DC`). Nodes should be a deep, professional blue (e.g., `#2C3E50`). Lines should initially be a subtle gray (e.g., `#B0C4DE`), transforming to a vibrant gold (e.g., `#FFD700`) or emerald green (e.g., `#2ECC71`) when simplified. Buttons should use contrasting but harmonious colors, such as a darker shade of blue or green.

#     User Actions:
#     1. Click and drag individual nodes within the display area to manually untangle the argument network, aiming to reduce line crossings and create clearer visual pathways.
#     2. Click the "Simplify Argument" button to trigger an automatic, animated re-arrangement of the nodes into a more organized and visually simplified structure.
#     Note:
#     - Your output should be implemented in JavaScript with HTML and CSS.
#     - Ensure that the output has both index.js and index.html files
#     """

#     dummy_feedback_2 = "add a relevant new green colour node. Separate the code into index.js and index.html files"
#     dummy_code = """<!DOCTYPE html>\n\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\"/>\n<meta content=\"width=device-width, initial-scale=1.0\" name=\"viewport\"/>\n<title>Referee Decision Simulator</title>\n<style>\n        body {\n            margin: 0;\n            display: flex;\n            justify-content: center;\n            align-items: center;\n            min-height: 100vh;\n            background: #1a1a1a;\n            font-family: Arial, sans-serif;\n        }\n        #container {\n            position: relative;\n        }\n        canvas {\n            border: 2px solid #444;\n        }\n        .controls {\n            position: absolute;\n            top: 20px;\n            left: 20px;\n            background: rgba(0,0,0,0.7);\n            padding: 15px;\n            border-radius: 8px;\n            color: white;\n        }\n        .var-button {\n            position: absolute;\n            bottom: 20px;\n            right: 20px;\n            padding: 10px 20px;\n            background: #006400;\n            color: white;\n            border: none;\n            border-radius: 4px;\n            cursor: pointer;\n        }\n        .decision-text {\n            position: absolute;\n            top: 20px;\n            right: 20px;\n            color: white;\n            background: rgba(0,0,0,0.7);\n            padding: 10px;\n            border-radius: 4px;\n            max-width: 200px;\n        }\n    </style>\n</head>\n<body>\n<div id=\"container\">\n<canvas id=\"field\"></canvas>\n<div class=\"controls\">\n<div>\n<label>Severity: </label>\n<input id=\"severity\" max=\"100\" min=\"0\" type=\"range\" value=\"50\"/>\n</div>\n<div>\n<label>Intentional: </label>\n<input id=\"intent\" type=\"checkbox\"/>\n</div>\n</div>\n<button class=\"var-button\" id=\"varButton\">VAR Review</button>\n<div class=\"decision-text\" id=\"decisionText\"></div>\n</div>\n<script src=\"index.js\"></script>\n<script>const canvas = document.getElementById('field');\nconst ctx = canvas.getContext('2d');\nconst severitySlider = document.getElementById('severity');\nconst intentCheckbox = document.getElementById('intent');\nconst varButton = document.getElementById('varButton');\nconst decisionText = document.getElementById('decisionText');\n\ncanvas.width = 800;\ncanvas.height = 600;\n\nconst field = {\n    zones: {\n        penaltyArea: { x: 600, y: 200, width: 200, height: 200 },\n        midfield: { x: 300, y: 200, width: 200, height: 200 },\n        wing: { x: 100, y: 100, width: 150, height: 400 }\n    },\n    referee: { x: 400, y: 300, animation: null, card: null },\n    particles: [],\n    varLines: []\n};\n\nfunction drawField() {\n    // Field background\n    ctx.fillStyle = '#2e8b57';\n    ctx.fillRect(0, 0, canvas.width, canvas.height);\n\n    // Field markings\n    ctx.strokeStyle = '#fff';\n    ctx.lineWidth = 2;\n    ctx.strokeRect(50, 50, canvas.width-100, canvas.height-100);\n    ctx.beginPath();\n    ctx.moveTo(canvas.width/2, 50);\n    ctx.lineTo(canvas.width/2, canvas.height-50);\n    ctx.stroke();\n\n    // Penalty areas\n    ctx.strokeRect(50, canvas.height/2-100, 150, 200);\n    ctx.strokeRect(canvas.width-200, canvas.height/2-100, 150, 200);\n}\n\nfunction drawReferee() {\n    ctx.fillStyle = '#000';\n    ctx.beginPath();\n    ctx.arc(field.referee.x, field.referee.y, 10, 0, Math.PI*2);\n    ctx.fill();\n\n    // Animation handling\n    if (field.referee.animation) {\n        const progress = Date.now() - field.referee.animation.start;\n        const duration = 1000;\n        \n        if (progress < duration) {\n            const angle = Math.sin(progress/200) * Math.PI/4;\n            const offsetY = Math.sin(progress/300) * 50;\n            \n            ctx.save();\n            ctx.translate(field.referee.x, field.referee.y);\n            ctx.rotate(angle);\n            \n            // Card drawing\n            ctx.fillStyle = field.referee.card;\n            ctx.fillRect(20, -15 + offsetY, 30, 40);\n            \n            // Particle effects\n            if (Math.random() < 0.3) {\n                field.particles.push({\n                    x: field.referee.x + 35,\n                    y: field.referee.y + offsetY,\n                    color: field.referee.card === 'yellow' ? '#ffd700' : '#ff0000',\n                    velocity: { x: (Math.random()-0.5)*2, y: -Math.random()*3 },\n                    life: 1\n                });\n            }\n            ctx.restore();\n        } else {\n            field.referee.animation = null;\n        }\n    }\n}\n\nfunction updateParticles() {\n    field.particles = field.particles.filter(p => {\n        p.x += p.velocity.x;\n        p.y += p.velocity.y;\n        p.velocity.y += 0.1;\n        p.life -= 0.02;\n        \n        ctx.fillStyle = p.color;\n        ctx.globalAlpha = p.life;\n        ctx.beginPath();\n        ctx.arc(p.x, p.y, 3, 0, Math.PI*2);\n        ctx.fill();\n        \n        return p.life > 0;\n    });\n    ctx.globalAlpha = 1;\n}\n\nfunction determineDecision(zone) {\n    const severity = severitySlider.value/100;\n    const intentional = intentCheckbox.checked;\n    \n    let decision = '';\n    \n    if (zone === 'penaltyArea') {\n        decision = severity > 0.7 ? 'RED CARD + PENALTY' : \n                  intentional ? 'YELLOW CARD + PENALTY' : 'PENALTY';\n    } else if (zone === 'midfield') {\n        decision = severity > 0.6 ? 'RED CARD' :\n                  (severity > 0.3 || intentional) ? 'YELLOW CARD' : 'FOUL';\n    } else if (zone === 'wing') {\n        decision = intentional ? 'YELLOW CARD' : 'FREE KICK';\n    }\n    \n    field.referee.animation = { start: Date.now() };\n    field.referee.card = decision.includes('RED') ? 'red' : \n                        decision.includes('YELLOW') ? 'yellow' : null;\n    \n    decisionText.textContent = `Decision: ${decision}\\n`\n        + `Severity: ${Math.round(severity*100)}%\\n`\n        + `Intent: ${intentional ? 'Deliberate' : 'Accidental'}`;\n}\n\ncanvas.addEventListener('click', (e) => {\n    const rect = canvas.getBoundingClientRect();\n    const x = e.clientX - rect.left;\n    const y = e.clientY - rect.top;\n\n    Object.entries(field.zones).forEach(([zone, area]) => {\n        if (x > area.x && x < area.x + area.width && \n            y > area.y && y < area.y + area.height) {\n            determineDecision(zone);\n            \n            // Zone highlight effect\n            ctx.strokeStyle = '#ff0';\n            ctx.lineWidth = 3;\n            ctx.setLineDash([10, 5]);\n            ctx.strokeRect(area.x, area.y, area.width, area.height);\n            ctx.setLineDash([]);\n        }\n    });\n});\n\nvarButton.addEventListener('click', () => {\n    // VAR line animation\n    field.varLines.push({\n        x1: canvas.width/2, y1: canvas.height/2,\n        x2: field.referee.x, y2: field.referee.y,\n        progress: 0\n    });\n});\n\nfunction animate() {\n    ctx.clearRect(0, 0, canvas.width, canvas.height);\n    drawField();\n    drawReferee();\n    updateParticles();\n    \n    // Animate VAR lines\n    field.varLines = field.varLines.filter(line => {\n        line.progress += 0.02;\n        ctx.strokeStyle = '#fff';\n        ctx.setLineDash([5, 3]);\n        ctx.beginPath();\n        ctx.moveTo(line.x1, line.y1);\n        ctx.lineTo(\n            line.x1 + (line.x2 - line.x1) * line.progress,\n            line.y1 + (line.y2 - line.y1) * line.progress\n        );\n        ctx.stroke();\n        return line.progress < 1;\n    });\n    ctx.setLineDash([]);\n    \n    requestAnimationFrame(animate);\n}\n\nanimate();</script></body>\n</html>"""
#     dummy_hf_request = HumanFeedbackRequest(
#         base_prompt=dummy_q_2,
#         base_code=str(dummy_code_answer),
#         miner_feedbacks=[
#             MinerFeedback(
#                 hotkey="0xfuck",
#                 miner_response_id="fuck",
#                 feedback=dummy_feedback_2,
#                 # feedback="Use colour to distinguish between intentional and unintentional fouls. Separate the code into index.js and index.html files.",
#             )
#         ],
#     )
#     import uuid

#     hf_id = str(uuid.uuid4())
#     res = await generate_human_feedback(dummy_hf_request, hf_id)
#     import json

#     with open("human_feedback_result.json", "w") as f:
#         json.dump(res, f, indent=2)


# if __name__ == "__main__":
#     asyncio.run(main())
