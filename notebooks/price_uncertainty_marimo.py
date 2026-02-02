"""
Price Uncertainty and Expected Value Optimization

A pedagogical demonstration of why expected value optimization can fail
under asymmetric price uncertainty, using battery storage as an example.
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from dataclasses import dataclass
    import numpy as np
    import matplotlib.pyplot as plt
    from pyscipopt import Model, quicksum
    return Model, dataclass, mo, np, plt, quicksum


@app.cell
def _(mo):
    mo.md("""
    # Price Uncertainty and Expected Value Optimization

    This notebook demonstrates a limitation of using MPC to optimise battery actions based on
    the expected value of the forecast price distribution. The key idea is that for the
    first action in the horizon, we will always know the price with certainty. This is because
    AEMO release the dispatch price for each 5 minute interval at the beginning of the interval,
    but they only make available a forecast (predispatch) price for the horizon into the future.
    So as we proceed in time we will get updates.

    This is critical. If we never received this updated dispatch price for the current interval,
    then the best we could do is to use the expected value of the price forecast. However,
    because we do receive these updates as we proceed forward in time, it is possible to pose better plans
    which trade-off opportunity and risk.

    While, I have not been able to formulate a way to solve this problem in general. This small example
    demonstrates one case where we can come up with a plan that is superior to the MPC solution based on
    the expected value.


    ## The Problem

    We consider a simplified version of the battery arbitrage problem.

    The battery can has three possible actions in each timestep
    - charge completely ($b_t^+=1, b_t^-=0$)
    - discharge completely ($b_t^+=0, b_t^-=-1$)
    - idle

    The state at each timestep is represented by $x \in [0,1]$
    with the state transition given by

    \[
        x_{t+1} = x_{t} + b^+_t + b^-_t
    \]

    the cost function is given by

    \[
        fC = \sum_t p_t(b^+_t + b^-_t) + r(b_t^-)
    \]

    where $p_t$ is the price in each timestep and $r$ is a static cost placed on charging the battery.

    THe inclusion of $r$ is important, without it the expected value optimisation turns out to be the best. While including $r$ may seem arbitrary, it is a simplified way to represent that needing to charge has other penalties. For a real battery these are in the form of
    - efficiency
    - network costs (charged on importing only usually)
    - degradation cost
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 1: Modeling Price Uncertainty

    We model uncertain prices using a **Probability Mass Function (PMF)** - a discrete
    probability distribution where each possible price has an associated probability.
    """)
    return


@app.cell
def _(dataclass, np):
    @dataclass
    class PMF:
        """
        Probability Mass Function for price forecasts.

        Represents a discrete distribution of possible prices,
        each with an associated probability.
        """
        values: list
        probabilities: list

        def __post_init__(self):
            if abs(sum(self.probabilities) - 1.0) > 1e-9:
                raise ValueError("Probabilities must sum to 1")
            if len(self.values) != len(self.probabilities):
                raise ValueError("Values and probabilities must have the same length")

        def sample(self):
            """Draw a random price according to the distribution."""
            return np.random.choice(self.values, p=self.probabilities)

        def expected_value(self):
            """Calculate the probability-weighted average price."""
            return sum(np.array(self.values) * np.array(self.probabilities))
    return (PMF,)


@app.cell
def _(mo):
    mo.md("""
    ## Part 2: The Carefully Constructed Price Scenario

    Here's where things get interesting. We'll construct price distributions that
    **deliberately expose the flaw** in expected value optimization.

    | Timestep | Price Distribution | Expected Value |
    |----------|-------------------|----------------|
    | 0 | $100 (certain) | $100 |
    | 1 | $900 (10%) or $10 (90%) | $99 |
    | 2 | $99 (certain) | $99 |

    ### The Trap

    An expected value optimizer sees:
    - Timestep 0: Expected price = $100 ← **Highest!**
    - Timestep 1: Expected price = $99
    - Timestep 2: Expected price = $99

    So it discharges at timestep 0 to capture the "highest" price.

    However, if we look at this in a risk vs reward manner then we would see the following

    Choosing not to discharge in step 0
    - Potential reward 10% chance of being able to sell at $900
    - Potential risk 90% chance that we end up waiting till timestep 3 to discharge which means we only make $99

    choosing to discharge in step 0
    - guarantee that we make $100

    So from a risk vs reward perspective it is better to not discharge in step 0. This we hard code into our alternative strategy.
    """)
    return


@app.cell
def _(PMF):
    # Define the price distributions
    price_distributions = {
        0: PMF(values=[100.0], probabilities=[1.0]),
        1: PMF(values=[900.0, 10.0], probabilities=[0.1, 0.9]),
        2: PMF(values=[99.0], probabilities=[1.0]),
    }
    return (price_distributions,)


@app.cell
def _(mo, price_distributions):
    # Show the expected values
    expected_values = {t: pmf.expected_value() for t, pmf in price_distributions.items()}
    mo.md(f"""
    ### Computed Expected Values

    - Timestep 0: **${expected_values[0]:.0f}**
    - Timestep 1: **${expected_values[1]:.0f}** (= 0.1 × $900 + 0.9 × $10)
    - Timestep 2: **${expected_values[2]:.0f}**

    Notice how timestep 1's expected value ($99) is almost the same as timestep 0 ($100),
    even though the *most likely* outcome at timestep 1 is only $10.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 3: Visualizing the Price Distributions

    Let's visualize these distributions to see the asymmetry more clearly.
    """)
    return


@app.cell
def _(plt, price_distributions):
    fig, axes = plt.subplots(1, len(price_distributions), figsize=(12, 4))

    for i, (timestep, pmf) in enumerate(price_distributions.items()):
        ax = axes[i]

        # Create horizontal bar chart
        colors = ['#e74c3c' if p < 0.5 else '#2ecc71' for p in pmf.probabilities]
        ax.barh(
            range(len(pmf.values)),
            pmf.probabilities,
            color=colors,
            height=0.6,
            alpha=0.8
        )

        # Add value labels
        for j, (val, prob) in enumerate(zip(pmf.values, pmf.probabilities)):
            ax.text(prob + 0.02, j, f'{prob*100:.0f}%', va='center', fontsize=11)

        ax.set_yticks(range(len(pmf.values)))
        ax.set_yticklabels([f'${v:.0f}' for v in pmf.values])
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Probability')
        ax.set_title(f'Timestep {timestep}\nE[price] = ${pmf.expected_value():.0f}')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Price Distributions at Each Timestep', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 4: The Battery Optimization Problem

    We have a simple battery with:
    - **Capacity**: 1 unit of energy (starts fully charged)
    - **Actions**: Charge (-1), Hold (0), or Discharge (+1) each timestep
    - **Charging cost**: $100 per unit charged (on top of energy price)
    - **Goal**: Minimize total cost (maximize revenue from discharging)

    We use **Model Predictive Control (MPC)**: at each timestep, we solve an
    optimization problem using the *expected* prices for future timesteps,
    then execute only the first action.
    """)
    return


@app.cell
def _():
    # Battery parameters
    STATIC_CHARGE_COST = 100  # Additional cost per unit charged
    return (STATIC_CHARGE_COST,)


@app.cell
def _(Model, STATIC_CHARGE_COST, quicksum):
    def expected_value_planner(price_series: list[float], current_state: float) -> list[float]:
        """
        MPC optimization based on expected value of prices.

        Uses SCIP solver to find optimal charge/discharge schedule.

        Args:
            price_series: List of (expected) prices for the planning horizon
            current_state: Current battery state of charge (0-1)

        Returns:
            List of optimal actions for each timestep
        """
        model = Model("Battery_Optimization")
        model.hideOutput()

        horizon = len(price_series)

        # Decision variables: charge and discharge actions
        battery_action_charge = [
            model.addVar(f"action_charge_{t}", vtype="I", lb=-1, ub=0)
            for t in range(horizon)
        ]
        battery_action_discharge = [
            model.addVar(f"action_discharge_{t}", vtype="I", lb=0, ub=1)
            for t in range(horizon)
        ]

        # State variables
        battery_state = [
            model.addVar(f"state_{t}", vtype="C", lb=0, ub=1)
            for t in range(horizon)
        ]

        # Binary indicators for charge/discharge
        is_negative = [model.addVar(vtype="B", name=f"is_negative_{t}") for t in range(horizon)]
        is_positive = [model.addVar(vtype="B", name=f"is_positive_{t}") for t in range(horizon)]

        # Indicator constraints
        for t in range(horizon):
            model.addConsIndicator(battery_action_discharge[t] >= 0, binvar=is_positive[t])
            model.addConsIndicator(battery_action_charge[t] <= 0, binvar=is_negative[t])

        # State transition constraints
        previous_state = current_state
        for t in range(horizon):
            model.addCons(
                battery_state[t] == previous_state + battery_action_discharge[t] + battery_action_charge[t]
            )
            previous_state = battery_state[t]

        # Objective: minimize cost (negative revenue)
        # Discharging earns price, charging costs price + static cost
        energy_cost = quicksum(
            battery_action_discharge[t] * price_series[t] +
            battery_action_charge[t] * (price_series[t] + STATIC_CHARGE_COST)
            for t in range(horizon)
        )

        model.setObjective(energy_cost, "minimize")
        model.optimize()

        return [
            model.getVal(battery_action_charge[t]) + model.getVal(battery_action_discharge[t])
            for t in range(horizon)
        ]
    return (expected_value_planner,)


@app.cell
def _(mo):
    mo.md("""
    ## Part 5: The Two Competing Strategies

    We'll compare two strategies:

    ### Strategy 1: Expected Value Optimization
    At each timestep, use MPC with expected prices to decide what to do.
    This is the "textbook" approach.

    ### Strategy 2: Wait-and-See (Alternate)
    **Don't discharge at timestep 0.** Instead, wait until timestep 1 when
    the uncertainty resolves, then optimize.

    The alternate strategy recognizes that the "high" expected price at timestep 1
    is misleading—it's better to wait and see what actually happens.
    """)
    return


@app.cell
def _(dataclass):
    @dataclass
    class Bot:
        """Tracks a strategy's state and actions during simulation."""
        state: float  # Current battery state of charge
        actions_taken: list  # History of actions
    return (Bot,)


@app.cell
def _(mo):
    mo.md("""
    ## Part 6: Single Trial Walkthrough

    Let's trace through one trial to see how the two strategies differ.
    """)
    return


@app.cell
def _(
    Bot,
    STATIC_CHARGE_COST,
    expected_value_planner,
    np,
    price_distributions,
):
    def simulate_single_trial(seed=None):
        """
        Simulate one trial, returning detailed step-by-step information.
        """
        if seed is not None:
            np.random.seed(seed)

        ev_bot = Bot(state=1.0, actions_taken=[])
        alt_bot = Bot(state=1.0, actions_taken=[])
        prices_realized = []
        steps_info = []

        for t in range(len(price_distributions)):
            # Sample the actual price for this timestep
            actual_price = price_distributions[t].sample()
            prices_realized.append(actual_price)

            # Build price series: actual current price + expected future prices
            future_expected = [
                price_distributions[k].expected_value()
                for k in range(t + 1, len(price_distributions))
            ]
            price_series = [actual_price] + future_expected

            # Expected Value Strategy: always optimize
            ev_actions = expected_value_planner(price_series, ev_bot.state)
            ev_bot.actions_taken.append(ev_actions[0])
            ev_bot.state += ev_actions[0]

            # Alternate Strategy: force hold at t=0, then optimize
            if t == 0:
                alt_actions = [0]
            else:
                alt_actions = expected_value_planner(price_series, alt_bot.state)
            alt_bot.actions_taken.append(alt_actions[0])
            alt_bot.state += alt_actions[0]

            steps_info.append({
                'timestep': t,
                'actual_price': actual_price,
                'price_series': price_series,
                'ev_action': ev_actions[0],
                'alt_action': alt_actions[0],
                'ev_state': ev_bot.state,
                'alt_state': alt_bot.state,
            })

        # Calculate final costs
        def calc_cost(actions, prices):
            cost = 0
            for action, price in zip(actions, prices):
                cost += action * price
                if action > 0:  # Charging
                    cost += abs(action) * STATIC_CHARGE_COST
            return cost

        ev_cost = calc_cost(ev_bot.actions_taken, prices_realized)
        alt_cost = calc_cost(alt_bot.actions_taken, prices_realized)

        return {
            'prices': prices_realized,
            'ev_actions': ev_bot.actions_taken,
            'alt_actions': alt_bot.actions_taken,
            'ev_cost': ev_cost,
            'alt_cost': alt_cost,
            'steps': steps_info,
        }
    return (simulate_single_trial,)


@app.cell
def _(mo, simulate_single_trial):
    # Run a single trial with a fixed seed for reproducibility
    example_trial = simulate_single_trial(seed=42)

    mo.md(f"""
    ### Example Trial Results

    **Prices realized**: {[f'${p:.0f}' for p in example_trial['prices']]}

    | Strategy | Actions | Final Cost |
    |----------|---------|------------|
    | Expected Value | {example_trial['ev_actions']} | ${example_trial['ev_cost']:.0f} |
    | Wait-and-See | {example_trial['alt_actions']} | ${example_trial['alt_cost']:.0f} |

    *Note: Positive action = charge, Negative = discharge, 0 = hold*
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 7: Monte Carlo Simulation

    One trial isn't enough—we need to see what happens on average across
    many different price realizations. Let's run 1000 trials.
    """)
    return


@app.cell
def _(mo, np, simulate_single_trial):
    # Run the Monte Carlo simulation
    np.random.seed(123)  # For reproducibility

    n_trials = 1000
    ev_costs = []
    alt_costs = []

    for _ in range(n_trials):
        trial_result = simulate_single_trial()
        ev_costs.append(trial_result["ev_cost"])
        alt_costs.append(trial_result["alt_cost"])

    ev_costs = np.array(ev_costs)
    alt_costs = np.array(alt_costs)

    mo.md(f"""
    ### Simulation Complete: {n_trials} trials

    | Metric | Expected Value Strategy | Wait-and-See Strategy |
    |--------|------------------------|----------------------|
    | Mean Cost | **${np.mean(ev_costs):.2f}** | **${np.mean(alt_costs):.2f}** |
    | Min Cost | ${np.min(ev_costs):.2f} | ${np.min(alt_costs):.2f} |
    | Max Cost | ${np.max(ev_costs):.2f} | ${np.max(alt_costs):.2f} |

    **Difference**: Wait-and-See saves **${np.mean(ev_costs) - np.mean(alt_costs):.2f}** on average
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
