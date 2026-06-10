# agent/cost_tracker.py
from dataclasses import dataclass, field

# Gemini 2.5 Flash Lite pricing (USD per 1M tokens)
COST_PER_M_INPUT = 0.10
COST_PER_M_OUTPUT = 0.40


@dataclass
class CostTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    loop_count: int = 0
    loop_costs: list = field(default_factory=list)

    def record_loop(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.loop_count += 1

        loop_cost = (
            (input_tokens / 1_000_000) * COST_PER_M_INPUT +
            (output_tokens / 1_000_000) * COST_PER_M_OUTPUT
        )
        self.loop_costs.append(loop_cost)

        print(
            f"[COST] Loop {self.loop_count}: "
            f"{input_tokens} input + {output_tokens} output tokens "
            f"= ${loop_cost:.6f}"
        )

    @property
    def total_cost(self) -> float:
        return sum(self.loop_costs)

    def summary(self) -> dict:
        return {
            "total_loops": self.loop_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "cost_per_loop_usd": round(self.total_cost / max(self.loop_count, 1), 6),
            "model": "gemini-2.5-flash-lite",
            "tier": "gemini-2.5-flash-lite",
        }