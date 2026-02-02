from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from pyscipopt import Model, quicksum
from tqdm import tqdm


static_charge_cost = 100

@dataclass
class PMF:
    """ Class for defining price forecast at a time interval"""
    values: list
    probabilities: list

    def __post_init__(self):
        if sum(self.probabilities) != 1:
            raise ValueError("Probabilities must sum to 1")

        if len(self.values) != len(self.probabilities):
            raise ValueError("Values and probabilities must have the same length")

    def sample(self):
        return np.random.choice(self.values, p=self.probabilities)

    def expected_value(self):
        return sum(np.array(self.values) * np.array(self.probabilities))




@dataclass
class bot:
    state: float
    actions_taken: list[float]


def expected_value_planner(price_series: list[float], current_state: float) -> list[float]:
    """
    MPC optimisation based on expected value of prices
    """
    model = Model("Battery_Optimization")
    model.hideOutput()

    horizon = len(price_series)

    battery_action_charge = [
        model.addVar(f"action_charge_{t}",
                     vtype="I",
                     lb=-1,
                     ub=0)
        for t in range(horizon)]

    battery_action_discharge = [
        model.addVar(f"action_discharge_{t}",
                     vtype="I",
                     lb=0,
                     ub=1)
        for t in range(horizon)
    ]

    battery_state = [
        model.addVar(
            f"state_{t}",
             vtype="C",
             lb=0,
             ub=1)
        for t in range(horizon)
    ]

    is_negative = [model.addVar(vtype="B", name=f"is_negative_{t}") for t in range(horizon)]
    is_positive = [model.addVar(vtype="B", name=f"is_positive_{t}") for t in range(horizon)]

    [model.addConsIndicator(
        battery_action_discharge[t] >= 0, binvar=is_positive[t]
    )  for t in range(horizon)]# negative flow must be zero
    [model.addConsIndicator(
        battery_action_charge[t] <= 0, binvar=is_negative[t]
    )  for t in range(horizon)] # positive flow must be zero


    previous_state = current_state
    for t in range(horizon):
        model.addCons(battery_state[t] == previous_state + battery_action_discharge[t] + battery_action_charge[t])
        previous_state = battery_state[t]

    energy_cost = quicksum(battery_action_discharge[t] * price_series[t] +
                           battery_action_charge[t] * (price_series[t] + static_charge_cost)
                           for t in range(horizon))

    model.setObjective(energy_cost, "minimize")

    # --- 6. Solve & Results ---
    model.optimize()

    return [model.getVal(battery_action_charge[t]) + model.getVal(battery_action_discharge[t]) for t in range(horizon)]


# define distributions of prices
# set up so that optimising with expected value will definitely discharge on first timestep
price_distributions = {
    0: PMF(values=[100.], probabilities=[1.]),
    1: PMF(values=[900., 10.], probabilities=[0.1, 0.9]),
    2: PMF(values=[99.], probabilities=[1.0])
}

# plot my price distributions
fig, axes = plt.subplots(1, len(price_distributions), figsize=(14, 6), sharex=True, sharey=True)
for i, (k, v) in enumerate(price_distributions.items()):
    ax = axes[i]

    ax.barh(v.values, v.probabilities, height=0.5 * np.array(v.values), color='red', align='center')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title(f"Timestep {k}")
    if i == 0:
        ax.set_ylabel("Price (Log)")
    ax.set_xlabel("Probability (Log)")


plt.tight_layout()
plt.show()

trials = 1000


def simulate_trial():
    expected_value_bot = bot(state=1, actions_taken=[])
    alternate_bot = bot(state=1, actions_taken=[])
    prices_realised = []


    for i in range(len(price_distributions)):
        current_interval_price = price_distributions[i].sample()
        future_interval_price_forecasts = [price_distributions[k].expected_value()
                                           for k in range(i+1, len(price_distributions))]

        price_series = [current_interval_price] + future_interval_price_forecasts

        if i==0:
            alternate_actions = [0]
        else:
            alternate_actions = expected_value_planner(price_series, alternate_bot.state)

        alternate_bot.actions_taken.append(alternate_actions[0])
        alternate_bot.state += alternate_actions[0]

        optimal_actions = expected_value_planner(price_series, expected_value_bot.state)
        expected_value_bot.actions_taken.append(optimal_actions[0])
        expected_value_bot.state += optimal_actions[0]


        prices_realised.append(current_interval_price)

    value_achieved_using_expected_value = (np.sum(np.array(prices_realised) * np.array(expected_value_bot.actions_taken))
                                           + np.sum(static_charge_cost * (np.array(expected_value_bot.actions_taken) > 0)))
    value_achieved_using_alternate_strategy = (np.sum(np.array(prices_realised) * np.array(alternate_bot.actions_taken))
                                               + np.sum(static_charge_cost * (np.array(alternate_bot.actions_taken) > 0)))

    return value_achieved_using_expected_value, value_achieved_using_alternate_strategy, prices_realised

results_optimising_for_expected_value = []
results_with_alternate_strategy = []
average_price = []
for i in tqdm(range(trials), desc="Simulating trials"):
    results = simulate_trial()
    results_optimising_for_expected_value.append(results[0])
    results_with_alternate_strategy.append(results[1])
    average_price.append(np.mean(results[2]))

print(f"Average cost achieved using expected value strategy: {np.mean(results_optimising_for_expected_value)}")
print(f"Average cost achieved using alternate strategy: {np.mean(results_with_alternate_strategy)}")