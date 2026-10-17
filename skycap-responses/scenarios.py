"""Scenarios for 03_drive.py: each drives one trajectory through skycap's Responses route and says what its
record should look like.

A scenario is an async function taking a ``Session``, registered with ``@scenario``. It talks to the model with
``session.respond(history)``, which applies the run's parameters (stream, sampling, reasoning, ...), and sends
replies back with ``session.replay(response)``, which applies the ``replay`` parameter. It states what it expects
by setting ``session.expect_paths``, ``session.expect_unbridged``, ``session.expect_finish_reasons`` (None: don't
check) and by ``session.check``.

To add one: write a function below, decorate it, and name it in a config's ``runs`` (or ``--scenario``).

Parameters every scenario reads through ``respond`` / ``replay`` (set in a config or with ``--set``):

    model               the served model name (default: $SKYCAP_MODEL)
    stream              true: stream, and read the final event; false: one JSON reply
    replay              verbatim | strip (no ids or statuses, as some clients send) |
                        drop_reasoning (a lossy client: expected to fork, so path checks are off)
    temperature, top_p, max_output_tokens, instructions, tool_choice, parallel_tool_calls,
    reasoning_effort    passed through as the Responses fields (skycap ignores some in token mode)
    extra_body          any other request fields, as an object
    headers             extra request headers, as an object
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai

TERMINAL = ("response.completed", "response.incomplete", "response.failed")


@dataclass
class Session:
    llm: openai.AsyncOpenAI
    params: dict[str, Any]
    created: dict[str, Any]
    http: httpx.AsyncClient
    server: str
    #: What the server says it runs (GET /healthz's ``capture``).
    capture: dict[str, Any]
    #: The paths ``finish`` should return (one per root-to-leaf path), or None not to check.
    expect_paths: int | None = 1
    #: Calls that should have re-rendered instead of extending the previous call, or None not to check.
    expect_unbridged: int | None = 0
    #: Each call's recorded finish_reason, in order (None: any), or None not to check.
    expect_finish_reasons: list[str | None] | None = None
    notes: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    replies: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text_mode(self) -> bool:
        """The server forwards to an upstream (vLLM's own /v1/responses) instead of sampling tokens itself."""
        return self.capture.get("mode") == "text"

    def p(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def check(self, ok: bool, what: str) -> None:
        if not ok:
            self.failed_checks.append(what)

    def request(self, history: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
        """The request body for ``history``: the run's parameters, then ``fields``."""
        body: dict[str, Any] = {}
        for name in ("instructions", "temperature", "top_p", "max_output_tokens", "tool_choice", "parallel_tool_calls"):
            if self.p(name) is not None:
                body[name] = self.p(name)
        if self.p("reasoning_effort") is not None:
            body["reasoning"] = {"effort": self.p("reasoning_effort")}
        body.update(self.p("extra_body") or {})
        body.update(fields)
        body["input"] = history
        return body

    async def respond(self, history: list[dict[str, Any]], *, headers: dict[str, str] | None = None, **fields: Any) -> dict[str, Any]:
        """One call; returns the response object as a dict (the final event's, when streamed)."""
        body = self.request(history, **fields)
        stream = bool(body.pop("stream", self.p("stream", False)))
        model = body.pop("model", self.p("model"))
        history = body.pop("input")
        extra_headers = {**(self.p("headers") or {}), **(headers or {})}
        if not stream:
            response = await self.llm.responses.create(
                model=model, input=history, extra_body=body, extra_headers=extra_headers
            )
            reply = response.model_dump(exclude_none=True)
        else:
            events = await self.llm.responses.create(
                model=model, input=history, stream=True, extra_body=body, extra_headers=extra_headers
            )
            reply, kinds = None, []
            async for event in events:
                kinds.append(event.type)
                if event.type in TERMINAL:
                    reply = event.response.model_dump(exclude_none=True)
            self.check(reply is not None, f"stream ended without a final event (saw {kinds[-3:]})")
            items, done = len((reply or {}).get("output", [])), kinds.count("response.output_item.done")
            if not self.text_mode:
                self.check(done == items, "one output_item.done per item")
            elif done != items:
                # Text mode relays the upstream's stream as sent. vLLM (0.31, Qwen3 parsers) closes an empty
                # message item between reasoning and tool calls that its final response leaves out; skycap
                # records the final response's items, which are what the client sends back.
                self.note(f"the upstream's stream closed {done} items for {items} in its final response")
            reply = reply or {}
        self.replies.append(reply)
        kinds = {item.get("type") for item in reply.get("output", [])}
        if "reasoning" in kinds and not kinds & {"message", "function_call"}:
            # The model ended its turn inside <think> (no </think>, or a cut-off there): the whole turn parses as
            # reasoning. renderers declines to bridge from such a turn, so the next call re-renders the history
            # (which Qwen3's template spells differently) and forks. Model behaviour, the same through chat.
            self.note(f"call {len(self.replies)} ended inside <think> with no answer; the next call can't bridge")
            self.expect_paths = self.expect_unbridged = None
        return reply

    def replay(self, reply: dict[str, Any]) -> list[dict[str, Any]]:
        """``reply``'s items as the client sends them back, per the ``replay`` parameter."""
        mode = self.p("replay", "verbatim")
        items = [dict(item) for item in reply.get("output", [])]
        if mode == "verbatim":
            return items
        if mode == "strip":
            return [{k: v for k, v in item.items() if k not in ("id", "status")} for item in items]
        if mode == "drop_reasoning":
            self.expect_paths = self.expect_unbridged = None
            return [item for item in items if item.get("type") != "reasoning"]
        raise ValueError(f"unknown replay mode {mode!r}")


Scenario = Callable[[Session], Awaitable[None]]
SCENARIOS: dict[str, Scenario] = {}


def scenario(run: Scenario) -> Scenario:
    SCENARIOS[run.__name__] = run
    return run


def refusal(error: openai.BadRequestError) -> str:
    """A 400's code and message, as skycap sent them."""
    body = error.body if isinstance(error.body, dict) else {}
    code = body.get("code") or error.code
    return f"{f'[{code}] ' if code else ''}{body.get('message') or error.message}"


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def text_of(reply: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in reply.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
    )


# -- tools the model can call; run here, on canned data ------------------------------------------------------
def _function(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(properties)},
    }


CITY_TIMES = {"paris": "14:05", "tokyo": "21:05", "new york": "08:05", "seoul": "21:05"}
FILES = {
    "README.md": "# demo\nA tiny project. Run `python main.py`.\n",
    "main.py": "def add(a, b):\n    return a - b  # bug\n\nprint(add(2, 3))\n",
}
TOOLS = {
    "get_time": (
        _function("get_time", "The current local time in a city.", {"city": {"type": "string"}}),
        lambda args: CITY_TIMES.get(str(args.get("city", "")).lower(), "unknown city"),
    ),
    "add": (
        _function("add", "Add two numbers.", {"a": {"type": "number"}, "b": {"type": "number"}}),
        lambda args: str(float(args.get("a", 0)) + float(args.get("b", 0))),
    ),
    "read_file": (
        _function("read_file", "Read a file of the project.", {"path": {"type": "string"}}),
        lambda args: FILES.get(str(args.get("path", "")), "no such file"),
    ),
    "exec_command": (
        _function("exec_command", "Run a shell command in the project and return its output.", {"cmd": {"type": "string"}}),
        lambda args: {"ls": "README.md\nmain.py\n", "python main.py": "-1\n"}.get(
            str(args.get("cmd", "")).strip(), f"sh: {args.get('cmd')}: simulated shell, command not available"
        ),
    ),
}


def tool_specs(*names: str) -> list[dict[str, Any]]:
    return [TOOLS[name][0] for name in names]


def run_tool(item: dict[str, Any]) -> dict[str, Any]:
    try:
        args = json.loads(item.get("arguments") or "{}")
        output = TOOLS[item["name"]][1](args if isinstance(args, dict) else {})
    except (KeyError, ValueError, TypeError) as error:
        output = f"error: {error}"
    return {"type": "function_call_output", "call_id": item["call_id"], "output": output}


async def tool_loop(session: Session, history: list[dict[str, Any]], tools: list[dict[str, Any]], steps: int) -> int:
    """Call, run every function call, send the results back; until a reply has no calls. Returns the calls made."""
    for step in range(1, steps + 1):
        reply = await session.respond(history, tools=tools)
        history += session.replay(reply)
        calls = [item for item in reply.get("output", []) if item.get("type") == "function_call"]
        if not calls:
            return step
        history += [run_tool(call) for call in calls]
    session.note(f"stopped after {steps} calls with tool calls still coming")
    return steps


# -- scenarios ----------------------------------------------------------------------------------------------------
@scenario
async def chat(session: Session) -> None:
    """Three user turns, no tools: every call after the first extends the previous one."""
    history = [user("Name a prime number between 10 and 20.")]
    for follow_up in ("Now one between 20 and 30.", "Which of the two is larger?", None):
        reply = await session.respond(history)
        session.check(reply.get("status") == "completed", f"status {reply.get('status')}")
        history += session.replay(reply)
        if follow_up:
            history.append(user(follow_up))


@scenario
async def tools(session: Session) -> None:
    """A tool loop (get_time, add), then a follow-up question: one path, every call bridged."""
    history = [user("What time is it in Paris and in Tokyo, and what is 17 + 25? Use the tools for all three.")]
    calls = await tool_loop(session, history, tool_specs("get_time", "add"), steps=6)
    called = sum(item.get("type") == "function_call" for reply in session.replies for item in reply.get("output", []))
    session.check(called > 0, "the model made no tool call")
    history.append(user("Thanks. Which city is later in the day?"))
    await session.respond(history, tools=tool_specs("get_time", "add"))
    session.note(f"{calls} calls in the loop, {called} tool calls")


CODEX_INSTRUCTIONS = (
    "You are a coding agent running in a terminal. You can run shell commands with exec_command and read files "
    "with read_file. Work step by step, verify your work, and finish with a short summary."
)


@scenario
async def codex_like(session: Session) -> None:
    """Codex's shape: instructions, two developer messages, an environment message, function tools, many steps.

    Exercises the developer fold: Qwen3's template has no developer role, so skycap merges the instructions and
    developer text into one leading system turn, on every call alike.
    """
    history = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "<permissions>sandbox: workspace-write; approvals: never</permissions>"}]},
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Keep answers short."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>cwd: /work/demo, shell: bash</environment_context>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "main.py prints the wrong sum. Find the bug and tell me the fix."}]},
    ]
    instructions = session.p("instructions") or CODEX_INSTRUCTIONS
    original = session.params
    session.params = {**original, "instructions": instructions}
    try:
        calls = await tool_loop(session, history, tool_specs("exec_command", "read_file"), steps=int(session.p("steps", 8)))
    finally:
        session.params = original
    session.note(f"{calls} calls")


@scenario
async def cutoff(session: Session) -> None:
    """A first answer cut off by max_output_tokens, then the conversation goes on.

    The status follows the server's --length-as-completed; the record says `length` either way. Path and bridge
    checks are off: a turn cut off inside its thinking may not bridge (SkyRL #2462).
    """
    session.expect_paths = session.expect_unbridged = None
    as_completed = bool(session.capture.get("length_as_completed"))
    history = [user("Explain in detail how a hash map works.")]
    reply = await session.respond(history, max_output_tokens=int(session.p("cutoff_tokens", 24)))
    want = "completed" if as_completed else "incomplete"
    session.check(reply.get("status") == want, f"cut-off status {reply.get('status')}, expected {want}")
    if not as_completed:
        session.check(
            (reply.get("incomplete_details") or {}).get("reason") == "max_output_tokens",
            f"incomplete_details {reply.get('incomplete_details')}",
        )
    history += session.replay(reply)
    history.append(user("Go on, briefly."))
    await session.respond(history)
    session.expect_finish_reasons = ["length", None]


@scenario
async def resample(session: Session) -> None:
    """The same request twice with no retry marker: two samples. One path if they came out identical, else two."""
    history = [user("Write one sentence about the sea.")]
    first = await session.respond(history)
    second = await session.respond(history)
    session.expect_paths = 1 if first.get("output") == second.get("output") else 2
    # skycap checks bridging on every call after a trajectory's first (`follows_a_call = bool(graph.nodes)`), and
    # the second call has no model turn to extend, so it is rendered and recorded as unbridged. Text mode never
    # bridges (it records no tokens), so it records none.
    session.expect_unbridged = 0 if session.text_mode else 1


@scenario
async def retry(session: Session) -> None:
    """The same request twice, the second marked as an SDK retry: it gets the first reply back, no new sample.

    Except a streamed call in text mode: skycap relays the upstream's stream as it arrives and keeps no copy to
    replay (skycap/retry.py), so the retry reaches the upstream and samples again, like a resample.
    """
    history = [user("Write one sentence about the mountains.")]
    first = await session.respond(history)
    second = await session.respond(history, headers={"x-stainless-retry-count": "1"})
    if session.text_mode and session.p("stream"):
        session.note("text mode replays no streamed reply: the retry sampled again")
        session.expect_paths = 1 if text_of(first) == text_of(second) else 2
        return
    session.check(first.get("id") == second.get("id"), "the retry got a different response")
    session.check(text_of(first) == text_of(second), "the retry's text differs")


@scenario
async def overflow(session: Session) -> None:
    """A prompt longer than the engine's context. Token mode: streamed, a response.failed with
    context_length_exceeded; not streamed, an HTTP 400; nothing recorded. Text mode: the upstream's own refusal,
    relayed as it sent it (its status and code are noted), and recorded as a failure."""
    session.expect_paths = 0
    session.expect_unbridged = None
    history = [user("word " * int(session.p("overflow_words", 60_000)))]
    if session.text_mode:
        try:
            reply = await session.respond(history)
        except openai.BadRequestError as error:
            session.note(f"upstream refused: 400 {refusal(error)}")
        else:
            session.check(reply.get("status") == "failed", f"an overflowing prompt was answered ({reply.get('status')})")
            session.note(f"upstream answered with a failed response: {reply.get('error')}")
        return
    if session.p("stream"):
        reply = await session.respond(history)
        error = reply.get("error") or {}
        session.check(reply.get("status") == "failed", f"status {reply.get('status')}")
        session.check(error.get("code") == "context_length_exceeded", f"error {error}")
        return
    try:
        await session.respond(history)
    except openai.BadRequestError as error:
        session.note(f"400 {refusal(error)}")
        session.check(error.code == "context_length_exceeded", f"code {error.code}")
    else:
        session.check(False, "an overflowing prompt was answered")


@scenario
async def refusals(session: Session) -> None:
    """What skycap refuses (HTTP 400) and what token mode ignores (answered), one request each.

    Text mode refuses only what the dialect can't map; what token mode refuses for itself (`truncation: auto`) or
    ignores goes to the upstream, which decides: its answer is noted, not checked.
    """
    session.expect_paths = session.expect_unbridged = None
    refused = {
        "previous_response_id": {"previous_response_id": "resp_x"},
        "custom tool": {"tools": [{"type": "custom", "name": "apply_patch"}]},
        "web_search tool": {"tools": [{"type": "web_search"}]},
        "truncation auto": {"truncation": "auto"},
        "item_reference": {"input": [{"type": "item_reference", "id": "x"}]},
        "encrypted reasoning": {
            "input": [user("q"), {"type": "reasoning", "summary": [], "encrypted_content": "gAAA"}, user("r")]
        },
    }
    ignored = {
        "tool_choice required": {"tool_choice": "required", "tools": tool_specs("add")},
        "parallel_tool_calls false": {"parallel_tool_calls": False, "tools": tool_specs("add")},
        "json schema": {"text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}}},
        "reasoning effort none": {"reasoning": {"effort": "none"}},
        "logprobs": {"include": ["message.output_text.logprobs"], "top_logprobs": 2},
    }
    forwarded: dict[str, dict[str, Any]] = {}
    if session.text_mode:
        forwarded = {"truncation auto": refused.pop("truncation auto"), **ignored}
        ignored = {}
    for what, fields in refused.items():
        history = fields.pop("input", [user("hi")])
        try:
            await session.respond(history, **fields)
        except openai.BadRequestError as error:
            session.note(f"{what}: 400 {refusal(error)}")
            continue
        session.check(False, f"{what} was answered, not refused")
    for what, fields in forwarded.items():
        try:
            reply = await session.respond([user("Say hi.")], **fields)
            session.note(f"{what}: forwarded; the upstream answered ({reply.get('status')})")
        except openai.BadRequestError as error:
            session.note(f"{what}: forwarded; the upstream refused: 400 {refusal(error)}")
    for what, fields in ignored.items():
        reply = await session.respond([user("Say hi.")], **fields)
        session.check(reply.get("status") == "completed", f"{what}: status {reply.get('status')}")
        if "tool_choice" in fields:
            session.check(reply.get("tool_choice") == "required", "tool_choice is echoed as sent")
