"""
linter.py:
  - enables use of ESLint library to lint input javascript code
  - will lint according to the rules specified in eslint.config.mjs
  - used in synthetic.py to trigger LLM queries to fix syntax errors when detected.
"""

import subprocess

from instructor import AsyncInstructor
from langfuse import observe
from loguru import logger
from pydantic import BaseModel, Field

from commons.llm import call_llm
from commons.types import CodeAnswer
from commons.utils import get_js_from_code_answer, log_to_langfuse


class LintResult(BaseModel):
    return_code: int = Field(
        description="status code returned from ESLint"
    )  # 1 if errors, 0 if no errors
    output: str = Field(description="stdout messages from ESLint")
    error: str = Field(description="stderr messages from ESLint")
    input: str = Field(description="input code passed to ESLint")


def setup_linting():
    """Set up the linting environment by ensuring ESLint is installed."""
    try:
        # Check if npm is available
        subprocess.run(["npm", "--version"], capture_output=True, check=True)

        # Install ESLint locally if not present
        subprocess.run(["npm", "install", "eslint"], capture_output=True, check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"Failed to set up linting environment: {e}")
        return False


def lint_code(code: str, id: str) -> LintResult:
    """
    calls ESLint on the input code and returns the result as a LintResult object.
    """
    try:
        # Check if eslint is installed
        npm_check = subprocess.run(
            ["npm", "list", "eslint"],
            capture_output=True,
            text=True,
            check=False,
        )
        if npm_check.returncode != 0:
            setup_linting()
        result = subprocess.run(
            [
                "npx",
                "eslint",
                "--quiet",  # only report errors, ignore warnings
                "--stdin",  # read from stdin instead of default behaviour of files
            ],
            input=code,
            capture_output=True,
            text=True,
            check=False,
        )

        return LintResult(
            return_code=result.returncode,
            output=result.stdout,
            error=result.stderr,
            input=code,
        )
    except Exception as e:
        logger.error(f"Error linting answer {id}: {e}")
        return LintResult(
            return_code=0,
            output="",
            error=str(e),
            input=code,
        )


@observe(as_type="generation", capture_input=True, capture_output=True)
async def _fix_syntax_errors(
    client: AsyncInstructor,
    model: str,
    answer: CodeAnswer,
    linter_feedback: str,
    id: str,
    attempt: int,
) -> CodeAnswer:
    """
    takes in a code answer and attempts to fix any syntax errors.
    """
    syntax_error_prompt = f"""
    <system>
        Here is some code that you must fix:
        <base_code>
            {get_js_from_code_answer(answer)}
        </base_code>
        Here is the error found in <base_code>:
        <error_message>
            {linter_feedback}
        </error_message>
        <role>
            You are an expert coding agent. You must modify <base_code> to fix the errors found in <error_message>.
        </role>
    </system>
    """
    messages = [
        {
            "role": "system",
            "content": syntax_error_prompt,
        },
    ]

    kwargs = {
        "response_model": CodeAnswer,
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 64000,
        "top_p": 0.1,
    }

    logger.error(f"{id}: fixing errors identified by linter, attempt {attempt}")
    try:
        result, completion = await call_llm(client, kwargs)
        log_to_langfuse(kwargs, result, completion)

        # update the original CodeAnswer with the fixed code
        for file in answer.files:
            if file.filename == "index.js":
                file.content = get_js_from_code_answer(result)
        return answer
    except Exception as e:
        # return original answer if failed to fix syntax errors
        logger.error(f"{id}: failed to fix errors: {e}")
        return answer


async def lint_and_fix_code(
    client: AsyncInstructor, model: str, answer: CodeAnswer, id: str, attempt: int = 1
) -> CodeAnswer:
    """
    @dev Executes ESlint on the input index.js file and will query LLM to fix any errors.
    @dev Will update the input answer object in-place with a fixed index.js file.
    @dev will recursively call itself 3 times to fix any syntax errors. Throw error on 3rd attempt.
    @param client: LLM Client object
    @param model: name of the LLM model used as a string
    @param answer: CodeAnswer object that is modified in-place
    @param qa_id: unique id for the code answer that is being modified.
    """
    # lint index.js, if there are errors (return_code is 1), then fix them with _fix_syntax_errors()
    js_code = get_js_from_code_answer(answer)
    lint_response = lint_code(js_code, id)
    if lint_response.return_code == 1:
        if attempt == 3:
            raise Exception(
                f"{id} failed to resolve linter errors after 3 attempts: {lint_response.output}"
            )
        # logger.info(f"{id} linter err: {lint_response.output}")
        # logger.info(f"{id} linter input: {lint_response.input}")
        fixed_answer = await _fix_syntax_errors(
            client, model, answer, lint_response.output, id, attempt
        )

        # lint the fixed answer
        return await lint_and_fix_code(
            client, model, fixed_answer, id, attempt=attempt + 1
        )

    # if linting was successful return corrected answer.
    elif lint_response.return_code == 0:
        if attempt > 1:
            logger.info(f"{id}: linter success after {attempt} attempts")
        return answer
    else:
        raise Exception(
            f"{id} unexpected linter return code: {lint_response.return_code} {lint_response.output}"
        )


# async def main():
#     """
#     main function used to for isolated testing of linter.py
#     """
#     import random

#     from commons.config import ANSWER_MODELS
#     from commons.llm import get_llm_api_client
#     from commons.types import CodeAnswer, FileObject

#     # Set up the linting environment
#     print("Setting up linting environment:", setup_linting())

#     client = get_llm_api_client()

#     # Create a test CodeAnswer with intentionally bad JavaScript code
#     bad_code = """const canvas = document.getElementById('canvas);\nconst ctx = canvas.getContext('2d');\nconst tooltip = document.getElementById(tooltip');\nconst chimeSound = new Audio('data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU...'); // Short base64 encoded chime sound\n\nlet width = canvas.width = window.innerWidth;\nlet height = canvas.height = window.innerHeight;\n\n// Colors\nconst colors = {\n  nightSky: '#0a0e23',\n  stars: '#f5f5f5',\n  candleBase: '#e6c229',\n  flameCenter: '#ffef5e',\n  flameEdge: '#ff9d00',\n  glow: 'rgba(255, 220, 100, 0.3)',\n  unlitCandle: '#3a3a4a',\n  prayerBubble: 'rgba(255, 255, 255, 0.9)'\n};\n\n// Prayer intentions\nconst prayers = [\n  \"Peace and healing\",\n  \"Strength in difficult times\",\n  \"Compassion for all\",\n  \"Gratitude for life's blessings\",\n  \"Hope for the future\",\n  \"Love to share with others\",\n  \"Patience and understanding\"\n];\n\n// State\nconst state = {\n  mainCandleLit: false,\n  daysLit: Array(7).fill(false),\n  stars: [],\n  particles: [],\n  mouseX: width / 2,\n  mouseY: height / 2,\n  flameAngle: 0,\n  lastFrameTime: 0\n};\n\n// Initialize stars\nfunction initStars() {\n  state.stars = [];\n  for (let i = 0; i < 100; i++) {\n    state.stars.push({\n      x: Math.random() * width,\n      y: Math.random() * height * 0.7,\n      size: Math.random() * 1.5 + 0.5,\n      opacity: Math.random() * 0.8 + 0.2,\n      speed: Math.random() * 0.02 + 0.01\n    });\n  }\n}\n\n// Draw night sky with stars\nfunction drawSky() {\n  // Gradient background\n  const gradient = ctx.createLinearGradient(0, 0, 0, height);\n  gradient.addColorStop(0, '#0a0e23');\n  gradient.addColorStop(1, '#1a1e33');\n  ctx.fillStyle = gradient;\n  ctx.fillRect(0, 0, width, height);\n\n  // Draw stars\n  ctx.fillStyle = colors.stars;\n  state.stars.forEach(star => {\n    star.opacity += (Math.random() - 0.5) * 0.1;\n    star.opacity = Math.max(0.1, Math.min(0.9, star.opacity));\n    ctx.globalAlpha = star.opacity;\n    ctx.beginPath();\n    ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);\n    ctx.fill();\n  });\n  ctx.globalAlpha = 1;\n}\n\n// Draw main candle\nfunction drawMainCandle() {\n  const candleX = width / 2;\n  const candleY = height * 0.3;\n  const candleWidth = 30;\n  const candleHeight = 100;\n\n  // Candle base\n  ctx.fillStyle = state.mainCandleLit ? colors.candleBase : colors.unlitCandle;\n  ctx.beginPath();\n  ctx.roundRect(candleX - candleWidth / 2, candleY, candleWidth, candleHeight, 5);\n  ctx.fill();\n\n  // Flame if lit\n  if (state.mainCandleLit) {\n    // Glow effect\n    const glow = ctx.createRadialGradient(\n      candleX, candleY, 0,\n      candleX, candleY, 100\n    );\n    glow.addColorStop(0, colors.glow);\n    glow.addColorStop(1, 'rgba(255, 220, 100, 0)');\n    ctx.fillStyle = glow;\n    ctx.beginPath();\n    ctx.arc(candleX, candleY, 100, 0, Math.PI * 2);\n    ctx.fill();\n\n    // Flame (responds to mouse)\n    const distance = Math.sqrt(\n      Math.pow(state.mouseX - candleX, 2) + \n      Math.pow(state.mouseY - candleY, 2)\n    );\n    const maxDistance = 200;\n    const influence = Math.max(0, 1 - distance / maxDistance);\n    const angle = Math.atan2(state.mouseY - candleY, state.mouseX - candleX);\n    state.flameAngle = angle * influence * 0.5;\n\n    const flameHeight = 50 + Math.sin(Date.now() * 0.01) * 5;\n    const flameWidth = 20 + Math.sin(Date.now() * 0.008) * 5;\n    \n    ctx.save();\n    ctx.translate(candleX, candleY);\n    ctx.rotate(state.flameAngle);\n    \n    // Flame gradient\n    const flameGradient = ctx.createLinearGradient(0, -flameHeight, 0, 0);\n    flameGradient.addColorStop(0, colors.flameEdge);\n    flameGradient.addColorStop(0.5, colors.flameCenter);\n    flameGradient.addColorStop(1, 'rgba(255, 255, 255, 0.7)');\n    \n    ctx.fillStyle = flameGradient;\n    ctx.beginPath();\n    ctx.moveTo(0, -flameHeight);\n    ctx.bezierCurveTo(\n      -flameWidth, -flameHeight * 0.7,\n      -flameWidth * 0.5, 0,\n      0, 0\n    );\n    ctx.bezierCurveTo(\n      flameWidth * 0.5, 0,\n      flameWidth, -flameHeight * 0.7,\n      0, -flameHeight\n    );\n    ctx.fill();\n    ctx.restore();\n\n    // Add occasional particles\n    if (Math.random() < 0.1) {\n      state.particles.push({\n        x: candleX + Math.sin(state.flameAngle) * 10,\n        y: candleY - flameHeight * 0.5,\n        size: Math.random() * 3 + 1,\n        life: 100,\n        vx: (Math.random() - 0.5) * 0.5,\n        vy: -Math.random() * 0.5 - 0.5,\n        color: Math.random() < 0.3 ? colors.flameCenter : colors.flameEdge\n      });\n    }\n  }\n}\n\n// Draw day candles\nfunction drawDayCandles() {\n  const candleSpacing = width / 8;\n  const startX = candleSpacing;\n  const candleY = height * 0.6;\n  const candleWidth = 15;\n  const candleHeight = 50;\n\n  for (let i = 0; i < 7; i++) {\n    const candleX = startX + i * candleSpacing;\n    \n    // Candle base\n    ctx.fillStyle = state.daysLit[i] ? colors.candleBase : colors.unlitCandle;\n    ctx.beginPath();\n    ctx.roundRect(candleX - candleWidth / 2, candleY, candleWidth, candleHeight, 3);\n    ctx.fill();\n\n    // Flame if lit\n    if (state.daysLit[i]) {\n      // Glow effect\n      const glow = ctx.createRadialGradient(\n        candleX, candleY, 0,\n        candleX, candleY, 50\n      );\n      glow.addColorStop(0, colors.glow);\n      glow.addColorStop(1, 'rgba(255, 220, 100, 0)');\n      ctx.fillStyle = glow;\n      ctx.beginPath();\n      ctx.arc(candleX, candleY, 50, 0, Math.PI * 2);\n      ctx.fill();\n\n      // Flame\n      const flameHeight = 30 + Math.sin(Date.now() * 0.01 + i) * 3;\n      const flameWidth = 12 + Math.sin(Date.now() * 0.008 + i) * 3;\n      \n      ctx.save();\n      ctx.translate(candleX, candleY);\n      \n      // Flame gradient\n      const flameGradient = ctx.createLinearGradient(0, -flameHeight, 0, 0);\n      flameGradient.addColorStop(0, colors.flameEdge);\n      flameGradient.addColorStop(0.5, colors.flameCenter);\n      flameGradient.addColorStop(1, 'rgba(255, 255, 255, 0.7)');\n      \n      ctx.fillStyle = flameGradient;\n      ctx.beginPath();\n      ctx.moveTo(0, -flameHeight);\n      ctx.bezierCurveTo(\n        -flameWidth, -flameHeight * 0.7,\n        -flameWidth * 0.5, 0,\n        0, 0\n      );\n      ctx.bezierCurveTo(\n        flameWidth * 0.5, 0,\n        flameWidth, -flameHeight * 0.7,\n        0, -flameHeight\n      );\n      ctx.fill();\n      ctx.restore();\n\n      // Add occasional particles\n      if (Math.random() < 0.05) {\n        state.particles.push({\n          x: candleX,\n          y: candleY - flameHeight * 0.5,\n          size: Math.random() * 2 + 1,\n          life: 80,\n          vx: (Math.random() - 0.5) * 0.3,\n          vy: -Math.random() * 0.3 - 0.3,\n          color: Math.random() < 0.3 ? colors.flameCenter : colors.flameEdge\n        });\n      }\n\n      // Check if mouse is hovering over this candle\n      const mouseOverCandle = Math.abs(state.mouseX - candleX) < 20 && \n                              Math.abs(state.mouseY - candleY) < 50;\n      \n      if (mouseOverCandle) {\n        // Draw prayer bubble\n        ctx.fillStyle = colors.prayerBubble;\n        ctx.beginPath();\n        ctx.roundRect(\n          candleX - 60, candleY - 80, \n          120, 40, 10\n        );\n        ctx.fill();\n        \n        // Draw triangle pointer\n        ctx.beginPath();\n        ctx.moveTo(candleX - 10, candleY - 40);\n        ctx.lineTo(candleX + 10, candleY - 40);\n        ctx.lineTo(candleX, candleY - 30);\n        ctx.fill();\n        \n        // Draw prayer text\n        ctx.fillStyle = '#333';\n        ctx.font = '14px Arial';\n        ctx.textAlign = 'center';\n        ctx.fillText(prayers[i], candleX, candleY - 60);\n      }\n    }\n  }\n}\n\n// Draw particles\nfunction drawParticles() {\n  state.particles = state.particles.filter(p => p.life > 0);\n  \n  state.particles.forEach(p => {\n    p.x += p.vx;\n    p.y += p.vy;\n    p.life -= 1;\n    \n    ctx.globalAlpha = p.life / 100;\n    ctx.fillStyle = p.color;\n    ctx.beginPath();\n    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);\n    ctx.fill();\n  });\n  \n  ctx.globalAlpha = 1;\n}\n\n// Handle mouse movement\nfunction handleMouseMove(e) {\n  state.mouseX = e.clientX;\n  state.mouseY = e.clientY;\n}\n\n// Handle clicks\nfunction handleClick(e) {\n  const x = e.clientX;\n  const y = e.clientY;\n  \n  // Check if main candle was clicked\n  const mainCandleX = width / 2;\n  const mainCandleY = height * 0.3;\n  const mainCandleWidth = 30;\n  const mainCandleHeight = 100;\n  \n  if (x > mainCandleX - mainCandleWidth / 2 && \n      x < mainCandleX + mainCandleWidth / 2 &&\n      y > mainCandleY && \n      y < mainCandleY + mainCandleHeight) {\n    state.mainCandleLit = !state.mainCandleLit;\n    playChime();\n    return;\n  }\n  \n  // Check if day candles were clicked\n  const candleSpacing = width / 8;\n  const startX = candleSpacing;\n  const candleY = height * 0.6;\n  const candleWidth = 15;\n  const candleHeight = 50;\n  \n  for (let i = 0; i < 7; i++) {\n    const candleX = startX + i * candleSpacing;\n    \n    if (x > candleX - candleWidth / 2 && \n        x < candleX + candleWidth / 2 &&\n        y > candleY && \n        y < candleY + candleHeight) {\n      if (!state.daysLit[i]) {\n        state.daysLit[i] = true;\n        playChime();\n      }\n      break;\n    }\n  }\n}\n\n// Play chime sound\nfunction playChime() {\n  chimeSound.currentTime = 0;\n  chimeSound.play().catch(e => console.log('Audio play failed:', e));\n}\n\n// Handle resize\nfunction handleResize() {\n  width = canvas.width = window.innerWidth;\n  height = canvas.height = window.innerHeight;\n  initStars();\n}\n\n// Animation loop\nfunction animate(timestamp) {\n  // Calculate delta time for frame-independent animation\n  const deltaTime = timestamp - state.lastFrameTime;\n  state.lastFrameTime = timestamp;\n  \n  // Clear canvas\n  ctx.clearRect(0, 0, width, height);\n  \n  // Draw elements\n  drawSky();\n  drawParticles();\n  drawDayCandles();\n  drawMainCandle();\n  \n  requestAnimationFrame(animate);\n}\n\n// Initialize\nfunction init() {\n  initStars();\n  \n  // Event listeners\n  window.addEventListener('resize', handleResize);\n  canvas.addEventListener('mousemove', handleMouseMove);\n  canvas.addEventListener('click', handleClick);\n  \n  // Start animation\n  requestAnimationFrame(animate);\n  \n  // Instructions\n  const instructions = document.createElement('div');\n  instructions.style.position = 'absolute';\n  instructions.style.bottom = '10px';\n  instructions.style.left = '10px';\n  instructions.style.color = 'white';\n  instructions.style.fontSize = '14px';\n  instructions.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';\n  instructions.style.padding = '5px 10px';\n  instructions.style.borderRadius = '5px';\n  instructions.innerHTML = 'Click candles to light them. Hover over lit candles to see prayer intentions.';\n  document.body.appendChild(instructions);\n}\n\n// Start when loaded\nwindow.addEventListener('load', init);"""

#     test_answer = CodeAnswer(
#         files=[FileObject(filename="index.js", content=bad_code, language="javascript")]
#     )

#     try:
#         # Test the linting and fixing functionality
#         _ = await lint_and_fix_code(
#             client=client,
#             model=random.choice(ANSWER_MODELS),
#             answer=test_answer,
#             id="test-run",
#         )

#     except Exception as e:
#         print(f"Error during testing: {e}")


# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(main())
