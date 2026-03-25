# SPDX-License-Identifier: MIT
"""
Agent Platform — permanent, non-versioned runtime for all GitHub project agents.

Import surface:

    from agent_platform.context import AgentContext
    from agent_platform.result import ActionResult, RunResult
    from agent_platform.summary import ExecutionSummary
    from agent_platform.errors import AgentError, DryRunViolation, ContractError
    from agent_platform.telemetry import log_event, begin_run, end_run
    from agent_platform.graphql_gateway import GraphQLGateway
    from agent_platform.github_gateway import GitHubGateway
    from agent_platform.runner import run_agent
"""
from agent_platform.context import AgentContext  # noqa: F401
from agent_platform.result import ActionResult, RunResult, ActionStatus  # noqa: F401
from agent_platform.summary import ExecutionSummary  # noqa: F401
from agent_platform.errors import AgentError, DryRunViolation, ContractError  # noqa: F401
from agent_platform.telemetry import log_event, begin_run, end_run  # noqa: F401
from agent_platform.graphql_gateway import GraphQLGateway  # noqa: F401
from agent_platform.github_gateway import GitHubGateway  # noqa: F401
from agent_platform.runner import run_agent  # noqa: F401

