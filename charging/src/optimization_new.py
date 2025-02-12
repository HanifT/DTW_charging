import numpy as np
import pandas as pd
import pyomo.environ as pyomo
from pyomo.util.infeasible import log_infeasible_constraints
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)


class EVCSP():

    def __init__(self, itinerary={}, itinerary_kwargs={}, inputs={}):
        self.sic = None  # ✅ Default value for infeasible cases
        if itinerary:
            self.inputs = self.ProcessItinerary(itinerary['trips'], **itinerary_kwargs)
        else:
            self.inputs = inputs

        if self.inputs:
            self.Build()

    def ProcessItinerary(self,
                         itinerary,
                         home_charger_likelihood=1,
                         work_charger_likelihood=0.75,
                         destination_charger_likelihood=.1,
                         consumption=782.928,  # J/meter
                         battery_capacity=60 * 3.6e6,  # J
                         initial_soc=.5,
                         final_soc=.5,
                         ad_hoc_charger_power=100e3,
                         home_charger_power=6.1e3,
                         work_charger_power=6.1e3,
                         destination_charger_power=100.1e3,
                         ac_dc_conversion_efficiency=.88,
                         max_soc=1,
                         min_soc=.2,
                         min_dwell_event_duration=15 * 60,
                         max_ad_hoc_event_duration=7.2e3,
                         min_ad_hoc_event_duration=5 * 60,
                         payment_penalty=60,
                         travel_penalty=1 * 60,
                         dwell_charge_time_penalty=0,
                         ad_hoc_charge_time_penalty=1,
                         tiles=7,
                         rng_seed=123,
                         residential_rate=None,
                         commercial_rate=None,
                         other_rate=0.42,  # Default value
                         **kwargs,
                         ):

        # 1. Create a random seed if none provided
        if not rng_seed:
            rng_seed = np.random.randint(1e6)
        rng = np.random.default_rng(rng_seed)

        # 2. We only pick ONE random day for the entire itinerary:
        random_day = rng.integers(1, 366)  # random integer day in 1..365
        random_day_sec = random_day * 86400

        # 3. Convert trip start times from HHMM to seconds-past-midnight
        start_times_hhmm = itinerary['STRTTIME'].to_numpy()
        end_times_hhmm = itinerary['ENDTIME'].to_numpy()
        start_times = (end_times_hhmm // 100) * 3600 + (end_times_hhmm % 100) * 60

        # 4. Driving / dwell durations
        durations = itinerary['TRVLCMIN'].copy().to_numpy()  # in minutes
        durations_sec = durations * 60
        dwell_times = itinerary['DWELTIME'].copy().to_numpy()
        dwell_times = np.where(dwell_times < 0, 1440, dwell_times)  # fix negative
        # dwell_times[dwell_times < 0] = dwell_times[dwell_times >= 0].mean()

        # 6. Expand trips/dwells by "tiles"
        trip_distances = np.tile(itinerary['TRPMILES'].to_numpy(), tiles) * 1609.34
        trip_discharge = trip_distances * consumption
        dwell_times_sec = dwell_times * 60
        dwells = np.tile(dwell_times_sec, tiles)  # dwell in seconds
        durations_sec = np.tile(durations_sec, tiles)

        # 7. Now build "absolute_start_times"
        #    We'll do it cumulatively for each trip.
        #    The FIRST trip is offset by random_day_sec + start_times[0].
        n_trips = len(dwells)

        # Extract month and day from the itinerary
        months = itinerary['Month'].to_numpy()
        days = itinerary['Day'].to_numpy()

        # Convert month/day to day of the year (DOY)
        # Considering leap year for simplicity (can adjust if needed)
        doy = (months - 1) * 30 + days  # Approximate DOY (adjust if exact calendar needed)

        # Calculate the absolute start time in seconds
        absolute_start_times = doy * 86400 + start_times  # DOY to seconds + HHMM in seconds

        # d_e_max_hours = np.array([int(np.ceil(dwells[e] / 3600)) for e in range(n_trips)], dtype=int)
        d_e_max_hours = (dwell_times // 60).astype(int)
        absolute_start_hours = (absolute_start_times // 3600).astype(int)

        # 8. Location type arrays
        location_types = np.tile(itinerary['WHYTRP1S'].to_numpy(), tiles)
        is_home = location_types == 1
        is_work = location_types == 10
        is_other = ((~is_home) & (~is_work))
        # 9. Charger assignment (like before)
        generator = np.random.default_rng(seed=rng_seed)
        destination_charger_power_array = np.zeros(n_trips)
        destination_charger_power_array[is_home] = home_charger_power
        destination_charger_power_array[is_work] = work_charger_power
        destination_charger_power_array[is_other] = destination_charger_power
        home_charger_selection = generator.random(is_home.sum()) <= home_charger_likelihood
        work_charger_selection = generator.random(is_work.sum()) <= work_charger_likelihood
        destination_charger_selection = generator.random(is_other.sum()) <= destination_charger_likelihood
        destination_charger_power_array[is_home] *= home_charger_selection
        destination_charger_power_array[is_work] *= work_charger_selection
        destination_charger_power_array[is_other] *= destination_charger_selection

        # 10. Build inputs
        inputs = {}
        inputs['end_times_hhmm'] = end_times_hhmm
        inputs['residential_rate'] = residential_rate
        inputs['commercial_rate'] = commercial_rate
        inputs['other_rate'] = other_rate
        inputs['n_e'] = n_trips
        inputs['s_i'] = initial_soc * battery_capacity
        inputs['s_f'] = final_soc * battery_capacity
        inputs['s_ub'] = max_soc * battery_capacity
        inputs['s_lb'] = min_soc * battery_capacity
        # Penalties
        inputs['c_db'] = np.ones(n_trips) * payment_penalty * is_other
        inputs['c_dd'] = np.ones(n_trips) * dwell_charge_time_penalty
        inputs['c_ab'] = np.ones(n_trips) * (travel_penalty + payment_penalty)
        inputs['c_ad'] = np.ones(n_trips) * ad_hoc_charge_time_penalty
        # Charger data
        inputs['r_d'] = destination_charger_power_array
        inputs['r_d_h'] = home_charger_power
        inputs['r_d_w'] = work_charger_power
        inputs['r_d_o'] = destination_charger_power
        inputs['r_a'] = np.ones(n_trips) * ad_hoc_charger_power
        # Time bounds for each event
        inputs['d_e_max'] = d_e_max_hours
        inputs['d_e_min'] = np.ones(n_trips) * min_dwell_event_duration
        inputs['a_e_min'] = np.ones(n_trips) * min_ad_hoc_event_duration
        inputs['a_e_max'] = np.ones(n_trips) * max_ad_hoc_event_duration
        # For the trip consumption
        inputs['d'] = trip_discharge
        inputs['b_c'] = battery_capacity
        inputs['l_i'] = trip_distances.sum()
        # location flags
        inputs['is_home'] = is_home
        inputs['is_work'] = is_work
        inputs['is_other'] = is_other
        # The final absolute start times
        inputs['absolute_start_times'] = absolute_start_hours
        # Store valid charging hours for each event
        inputs['valid_hours'] = {
            e: list(range(absolute_start_hours[e], absolute_start_hours[e] + max(1, d_e_max_hours[e]) + 1))
            for e in range(n_trips)
        }

        if isinstance(inputs['residential_rate'], list) and len(inputs['residential_rate']) == 8760:
            mean_res_rate = np.mean(inputs['residential_rate'])
            inputs['residential_rate'].append(mean_res_rate)  # Add hour 8760

        if isinstance(inputs['commercial_rate'], list) and len(inputs['commercial_rate']) == 8760:
            mean_com_rate = np.mean(inputs['commercial_rate'])
            inputs['commercial_rate'].append(mean_com_rate)  # Add hour 8760

        # If needed, transform the rate arrays from dict to list
        if isinstance(inputs['residential_rate'], dict) and "rate" in inputs['residential_rate']:
            inputs['residential_rate'] = [inputs['residential_rate']["rate"][str(hour)] for hour in range(8761)]

        if isinstance(inputs['commercial_rate'], dict) and "rate" in inputs['commercial_rate']:
            inputs['commercial_rate'] = [inputs['commercial_rate']["rate"][str(hour)] for hour in range(8761)]

        # for e in range(n_trips):
        #     print(f"Event {e}: Start Hour = {absolute_start_hours[e]}, End Hour = {absolute_start_hours[e] + d_e_max_hours[e]},dwell_time = {d_e_max_hours[e]},  Valid Hours = {inputs['valid_hours'][e]}")
        #
        # for e in range(n_trips):
        #     print(f"Event {e}: dwell times = {d_e_max_hours[e]}")
        return inputs

    def Solve(self, solver_kwargs={}):
        solver = pyomo.SolverFactory(**solver_kwargs)
        # Add MIP gap for CBC (set to 5%)
        solver.options['ratio'] = 0.20  # 5% MIP gap
        solver.options['threads'] = 20
        solver.options['maxMemory'] = 48000
        solver.options['loglevel'] = 2  # Verbose output
        solver.options['infeasibility'] = 'on'  # Show infeasibility report if supported
        # Add heuristic approach (optional)
        solver.options['heuristics'] = 'on'  # Enable solver's internal heuristic
        res = solver.solve(self.model)
        self.solver_status = res.solver.status
        self.solver_termination_condition = res.solver.termination_condition

        self.Solution()
        self.Compute


    def Build(self):

        # Pulling the keys from the inputs dict
        keys = self.inputs.keys()
        # Initializing the model as a concrete model
        # (as in one that has fixed inputted values)
        self.model = pyomo.ConcreteModel(name="EVCSP_Model")
        # Adding variables
        self.Variables()
        # Upper-level objective (inconvenience)
        self.UpperObjective()  # New
        # Lower-level KKT conditions for cost minimization
        self.LowerKKT()  # New
        # Bounds constraints
        self.Bounds()
        # Unit commitment constraints
        self.Unit_Commitment()

    def Variables(self):
        self.model.E = pyomo.Set(initialize=range(self.inputs['n_e']))

        # Destination charging
        # You can do 0..8759 or only the hours that appear in your trips.
        EH = [(e, h) for e in self.model.E for h in self.inputs['valid_hours'][e]]
        self.model.EH = pyomo.Set(initialize=EH, dimen=2)

        self.model.x_d = pyomo.Var(self.model.EH, domain=pyomo.NonNegativeReals)
        self.model.x_a = pyomo.Var(self.model.EH, domain=pyomo.NonNegativeReals)

    def UpperObjective(self):
        # Build a dictionary "penalty_for_event[e]" = 0 if home/work, else some penalty
        penalty_for_event = {e: self.inputs['c_ab'][e] for e in self.model.E}
        penalty_for_charging = {e: self.inputs['c_ad'][e] for e in self.model.E }
        # Updated objective: sum over (e, h) in EH
        def inconvenience_expression(m):
            return sum(
                (penalty_for_event[e] * m.x_a[e, h]) + (penalty_for_charging[e] * m.x_a[e, h])

                # Inconvenience applies only to x_a (ad-hoc)
                for (e, h) in m.EH
            )

        # Create the Pyomo objective
        self.model.upper_objective = pyomo.Objective(
            expr=inconvenience_expression(self.model),
            sense=pyomo.minimize
        )

    def LowerKKT(self):
        model = self.model

        # 1) Dual variables for x_d
        model.lambda_d_min = pyomo.Var(model.EH, domain=pyomo.NonNegativeReals)
        model.lambda_d_max = pyomo.Var(model.EH, domain=pyomo.NonNegativeReals)
        model.lambda_a_min = pyomo.Var(model.EH, domain=pyomo.NonNegativeReals)
        model.lambda_a_max = pyomo.Var(model.EH, domain=pyomo.NonNegativeReals)

        # 2) Stationarity for x_d
        def stationarity_xd_rule(m, e, h):
            h_capped = min(h, 8760)  # Ensure h doesn't exceed 8760 (max hourly index for a year)

            if isinstance(self.inputs['residential_rate'], list):
                cost_rate_home = self.inputs['residential_rate'][h_capped]
            else:
                cost_rate_home = self.inputs['residential_rate']

            if isinstance(self.inputs['commercial_rate'], list):
                cost_rate_work = self.inputs['commercial_rate'][h_capped]
            else:
                cost_rate_work = self.inputs['commercial_rate']

            if isinstance(self.inputs['other_rate'], list):
                cost_rate_other = self.inputs['other_rate'][h_capped]
            else:
                cost_rate_other = self.inputs['other_rate']

            cost_rate = (
                    cost_rate_home * self.inputs['is_home'][e]
                    + cost_rate_work * self.inputs['is_work'][e]
                    + cost_rate_other * self.inputs['is_other'][e]
            )

            return cost_rate - m.lambda_d_min[e, h] + m.lambda_d_max[e, h] == 0

        model.stationarity_xd = pyomo.Constraint(model.EH, rule=stationarity_xd_rule)
        # 3) Stationarity for x_a
        def stationarity_xa_rule(m, e, h):
            if isinstance(self.inputs['other_rate'], list):
                cost_rate = self.inputs['other_rate'][h]
            else:
                cost_rate = self.inputs['other_rate']

            return cost_rate - m.lambda_a_min[e, h] + m.lambda_a_max[e, h] == 0

        model.stationarity_xa = pyomo.Constraint(model.EH, rule=stationarity_xa_rule)
        # 4) Primal feasibility: 0 <= x_d[e,h] <= r_d[e]
        def xd_min_rule(m, e, h):
            return m.x_d[e, h] >= 0


        def xd_max_rule(m, e, h):
            if self.inputs['is_home'][e]:
                max_power = min(self.inputs['r_d'][e], self.inputs['r_d_h'])
            elif self.inputs['is_work'][e]:
                max_power = min(self.inputs['r_d'][e], self.inputs['r_d_w'])
            elif self.inputs['is_other'][e]:
                max_power = min(self.inputs['r_d'][e], self.inputs['r_d_o'])
            else:
                max_power = 0  # No charging if undefined

            return m.x_d[e, h] <= max_power

        model.xd_min_con = pyomo.Constraint(model.EH, rule=xd_min_rule)
        model.xd_max_con = pyomo.Constraint(model.EH, rule=xd_max_rule)

        # Similarly for x_a: 0 <= x_a[e,h] <= r_a[e]
        def xa_min_rule(m, e, h):
            return m.x_a[e, h] >= 0

        def xa_max_rule(m, e, h):
            return m.x_a[e, h] <= self.inputs['r_a'][e]

        model.xa_min_con = pyomo.Constraint(model.EH, rule=xa_min_rule)
        model.xa_max_con = pyomo.Constraint(model.EH, rule=xa_max_rule)

    def Bounds(self):
        model = self.model
        # Create a ConstraintList to store SoC constraints
        model.bounds = pyomo.ConstraintList()
        # SoC variables for each event
        model.ess_state = pyomo.Var(model.E,
                                    domain=pyomo.NonNegativeReals,
                                    bounds=(self.inputs['s_lb'], self.inputs['s_ub']))
        model.soc_after_trip = pyomo.Var(model.E,
                                         domain=pyomo.NonNegativeReals,
                                         bounds=(self.inputs['s_lb'], self.inputs['s_ub']))
        # 1. Initial SOC at event e=0
        model.bounds.add(model.ess_state[0] >= 0.02 * self.inputs['s_i'])

        # 2. For each event e, define SoC after trip & SoC after charging
        for e in model.E:
            if e == 0:
                model.bounds.add(
                    model.soc_after_trip[e] == self.inputs['s_i'] - self.inputs['d'][e]
                )
            else:
                model.bounds.add(
                    model.soc_after_trip[e] == model.ess_state[e - 1] - self.inputs['d'][e]
                )
            # (B) SoC at the end of event e = after_trip[e] + total charged energy in event e
            model.bounds.add(
                model.ess_state[e] == model.soc_after_trip[e] + sum((model.x_d[e, h] + model.x_a[e, h]) for (e2, h) in model.EH if e2 == e)
            )
        # 3. Ensure the final SOC is at least or exactly the required final SoC
        model.bounds.add(model.ess_state[self.inputs['n_e'] - 1] == self.inputs['s_f'])


    def Unit_Commitment(self):
        model = self.model

        # Pre-calculate valid hours for each event e
        valid_hours = {}
        for e in model.E:
            t_start = int(self.inputs['absolute_start_times'][e])
            t_end = int((self.inputs['absolute_start_times'][e] + self.inputs['d_e_max'][e]))
            valid_hours[e] = range(t_start, t_end + 1)

        # 1) Zero out x_d[e,h] if hour h not in the dwell window
        def no_charge_outside_window_d_rule(model, e, h):
            if h not in self.inputs['valid_hours'][e]:
                return model.x_d[e, h] == 0
            return pyomo.Constraint.Skip

        model.no_charge_outside_window_d = pyomo.Constraint(model.EH, rule=no_charge_outside_window_d_rule)

        def no_charge_outside_window_a_rule(model, e, h):
            if h not in self.inputs['valid_hours'][e]:
                return model.x_a[e, h] == 0
            return pyomo.Constraint.Skip

        model.no_charge_outside_window_a = pyomo.Constraint(model.EH, rule=no_charge_outside_window_a_rule)
        # 2) Mutual Exclusivity of Charging Modes

        def mutual_exclusivity_rule(model, e, h):
            # Ensure that x_d and x_a do not operate simultaneously beyond the charger’s capacity
            return model.x_d[e, h] + model.x_a[e, h] <= max(self.inputs['r_d'][e], self.inputs['r_a'][e])

        # Add the constraint to the model
        model.mutual_exclusivity = pyomo.Constraint(model.EH, rule=mutual_exclusivity_rule)


    def Solution(self):
        model_vars = self.model.component_map(ctype=pyomo.Var)
        serieses = []  # collection to hold the converted "serieses"
        for k in model_vars.keys():  # this is a map of {name:pyo.Var}
            v = model_vars[k]

            # make a pd.Series from each
            s = pd.Series(v.extract_values(), index=v.extract_values().keys())

            # if the series is multi-indexed we need to unstack it...
            if type(s.index[0]) == tuple:  # it is multi-indexed
                s = s.unstack(level=1)
            else:
                s = pd.DataFrame(s)  # force transition from Series -> df
            s.columns = pd.MultiIndex.from_tuples([(k, t) for t in s.columns])
            serieses.append(s)
        self.solution = pd.concat(serieses, axis=1)

    @property
    def Compute(self):

        def seconds_to_hhmm(seconds):
            hours = (seconds // 3600) % 24
            minutes = (seconds % 3600) // 60
            return int(hours * 100 + minutes)

        soc_values = np.array([pyomo.value(self.model.ess_state[e]) for e in self.model.E]) / self.inputs['b_c']
        soc_after_trip_values = np.array([pyomo.value(self.model.soc_after_trip[e]) for e in self.model.E]) / self.inputs['b_c']

        try:
            soc_values = np.array([pyomo.value(self.model.ess_state[e]) for e in self.model.E]) / self.inputs['b_c']
        except ValueError:
            print(" ess_state is uninitialized due to infeasibility. Skipping this itinerary.")
            return  # Skip further processing for this infeasible case

        # Prepend initial SOC
        soc = np.insert(soc_values, 0, self.inputs['s_i'] / self.inputs['b_c'])

        # Store SoC in the solution DataFrame
        self.solution[("soc", 0)] = soc[1:]  # SOC after charging
        self.solution[("soc_start_trip", 0)] = soc[:-1]  # SOC before the trip
        self.solution[("soc_end_trip", 0)] = soc_after_trip_values  # SOC after trip


        # 2) Compute the "inconvenience" metric (SIC)
        total_inconvenience = 0.0
        for e in self.model.E:
            penalty_e = self.inputs['c_db'][e] if self.inputs['is_other'][e] else 0.0

            # Valid charging hours for the event
            start_hour = int(self.inputs['absolute_start_times'][e])
            end_hour = int(start_hour + int(self.inputs['d_e_max'][e]))

            for h in range(start_hour, end_hour + 1):
                xd_val = pyomo.value(self.model.x_d[e, h])
                total_inconvenience += penalty_e * xd_val

        self.sic = (total_inconvenience / 60) / (self.inputs['l_i'] / 1e3)

        # Extract start times and durations from itinerary

        end_times_hhmm = self.inputs.get('end_times_hhmm')
        end_times_hhmm = np.array(end_times_hhmm)
        end_times = (end_times_hhmm // 100) * 3600 + (end_times_hhmm % 100) * 60

        # 3) Calculate cost & total kWh for each event
        charging_costs = []
        charging_kwh_total = []
        charging_start_times = []
        charging_end_times = []
        hourly_charging_details = []

        for e in self.model.E:
            total_energy_j = 0.0
            total_cost_e = 0.0
            charging_start_sec = end_times[e]
            # ✅ Charging starts exactly at the end of the trip
            # charging_start_sec = end_times[e]

            # Find the last time charging occurred
            last_charging_sec = None
            first_charging_sec = None
            hourly_charging = {}

            # Calculate total charged energy and cost
            for h in self.inputs['valid_hours'][e]:

                xd_val = pyomo.value(self.model.x_d[e, h])
                xa_val = pyomo.value(self.model.x_a[e, h])
                event_energy_h = (xd_val + xa_val) * 2.7778e-7
                total_energy_j += event_energy_h

                if event_energy_h > 1e-9:
                    if first_charging_sec is None:
                        first_charging_sec = h * 3600
                    # Use power in kW (NOT converted to kWh) to get the correct duration in seconds
                    hour_key = f"{h % 24}:00"  # Keep the full hour for clarity (e.g., 1525:00)

                    if hour_key in hourly_charging:
                        hourly_charging[hour_key] += round(event_energy_h, 3)
                    else:
                        hourly_charging[hour_key] = round(event_energy_h, 3)

                    last_charging_sec = h * 3600 + (event_energy_h / max(self.inputs['r_d'][e], self.inputs['r_a'][e]))

                h_capped = min(h, 8760)

                if self.inputs['is_home'][e]:
                    rate = self.inputs['residential_rate'][h_capped] if isinstance(self.inputs['residential_rate'], list) else self.inputs['residential_rate']
                elif self.inputs['is_work'][e]:
                    rate = self.inputs['commercial_rate'][h_capped] if isinstance(self.inputs['commercial_rate'], list) else self.inputs['commercial_rate']
                else:
                    rate = self.inputs['other_rate']

                total_cost_e += (event_energy_h * rate)

            # Convert total energy to kWh
            total_kwh = total_energy_j
            charging_kwh_total.append(total_kwh)
            charging_costs.append(total_cost_e)
            hourly_charging_details.append(hourly_charging)

            # Calculate start and end times only if there's charging
            if total_kwh > 0:
                start_sec = charging_start_sec if first_charging_sec is None else first_charging_sec
                end_sec = last_charging_sec if last_charging_sec is not None else start_sec

                charging_start_times.append(seconds_to_hhmm(start_sec))
                charging_end_times.append(seconds_to_hhmm(end_sec))
            else:
                charging_start_times.append(np.nan)
                charging_end_times.append(np.nan)

        # Store final results
        self.solution[("charging_kwh_total", 0)] = charging_kwh_total
        self.solution[("charging_cost", 0)] = charging_costs
        self.solution[("charging_start_time_HHMM", 0)] = charging_start_times
        self.solution[("charging_end_time_HHMM", 0)] = charging_end_times
        self.solution[("hourly_charging_details", 0)] = hourly_charging_details


class EVCSP_delay():

    def __init__(self, itinerary={}, itinerary_kwargs={}, inputs={}):
        self.sic = None  # ✅ Default value for infeasible cases
        if itinerary:
            self.inputs = self.ProcessItinerary(itinerary['trips'], **itinerary_kwargs)
        else:
            self.inputs = inputs

        if self.inputs:
            self.Build()

    def ProcessItinerary(self,
                         itinerary,
                         home_charger_likelihood=1,
                         work_charger_likelihood=0.75,
                         destination_charger_likelihood=.1,
                         midnight_charging_prob=0.5,
                         consumption=782.928,  # J/meter
                         battery_capacity=60 * 3.6e6,  # J
                         initial_soc=.5,
                         final_soc=.5,
                         ad_hoc_charger_power=100e3,
                         home_charger_power=6.1e3,
                         work_charger_power=6.1e3,
                         destination_charger_power=100.1e3,
                         ac_dc_conversion_efficiency=.88,
                         max_soc=1,
                         min_soc=.2,
                         min_dwell_event_duration=15 * 60,
                         max_ad_hoc_event_duration=7.2e3,
                         min_ad_hoc_event_duration=5 * 60,
                         payment_penalty=60,
                         travel_penalty=1 * 60,
                         time_penalty= 60,
                         dwell_charge_time_penalty=0,
                         ad_hoc_charge_time_penalty=1,
                         tiles=7,
                         rng_seed=123,
                         residential_rate=None,
                         commercial_rate=None,
                         other_rate=0.42,  # Default value
                         **kwargs,
                         ):

        # 1. Create a random seed if none provided
        if not rng_seed:
            rng_seed = np.random.randint(1e6)
        rng = np.random.default_rng(rng_seed)

        # 2. We only pick ONE random day for the entire itinerary:
        random_day = rng.integers(1, 366)  # random integer day in 1..365
        random_day_sec = random_day * 86400
        allow_midnight_charging = rng.random(len(itinerary)) < midnight_charging_prob
        # 3. Convert trip start times from HHMM to seconds-past-midnight
        start_times_hhmm = itinerary['STRTTIME'].to_numpy()
        end_times_hhmm = itinerary['ENDTIME'].to_numpy()
        start_times = (end_times_hhmm // 100) * 3600 + (end_times_hhmm % 100) * 60

        # 4. Driving / dwell durations
        durations = itinerary['TRVLCMIN'].copy().to_numpy()  # in minutes
        durations_sec = durations * 60
        dwell_times = itinerary['DWELTIME'].copy().to_numpy()
        dwell_times = np.where(dwell_times < 0, 1440, dwell_times)  # fix negative
        # dwell_times[dwell_times < 0] = dwell_times[dwell_times >= 0].mean()

        # 6. Expand trips/dwells by "tiles"
        trip_distances = np.tile(itinerary['TRPMILES'].to_numpy(), tiles) * 1609.34
        trip_discharge = trip_distances * consumption
        dwell_times_sec = dwell_times * 60
        dwells = np.tile(dwell_times_sec, tiles)  # dwell in seconds
        durations_sec = np.tile(durations_sec, tiles)

        # 7. Now build "absolute_start_times"
        #    We'll do it cumulatively for each trip.
        #    The FIRST trip is offset by random_day_sec + start_times[0].
        n_trips = len(dwells)

        # Extract month and day from the itinerary
        months = itinerary['Month'].to_numpy()
        days = itinerary['Day'].to_numpy()

        # Convert month/day to day of the year (DOY)
        # Considering leap year for simplicity (can adjust if needed)
        doy = (months - 1) * 30 + days  # Approximate DOY (adjust if exact calendar needed)

        # Calculate the absolute start time in seconds
        absolute_start_times = doy * 86400 + start_times  # DOY to seconds + HHMM in seconds

        # d_e_max_hours = np.array([int(np.ceil(dwells[e] / 3600)) for e in range(n_trips)], dtype=int)
        d_e_max_hours = (dwell_times // 60).astype(int)
        absolute_start_hours = (absolute_start_times // 3600).astype(int)

        # 8. Location type arrays
        location_types = np.tile(itinerary['WHYTRP1S'].to_numpy(), tiles)
        is_home = location_types == 1
        is_work = location_types == 10
        is_other = ((~is_home) & (~is_work))
        # 9. Charger assignment (like before)
        generator = np.random.default_rng(seed=rng_seed)
        destination_charger_power_array = np.zeros(n_trips)
        destination_charger_power_array[is_home] = home_charger_power
        destination_charger_power_array[is_work] = work_charger_power
        destination_charger_power_array[is_other] = destination_charger_power
        home_charger_selection = generator.random(is_home.sum()) <= home_charger_likelihood
        work_charger_selection = generator.random(is_work.sum()) <= work_charger_likelihood
        destination_charger_selection = generator.random(is_other.sum()) <= destination_charger_likelihood
        destination_charger_power_array[is_home] *= home_charger_selection
        destination_charger_power_array[is_work] *= work_charger_selection
        destination_charger_power_array[is_other] *= destination_charger_selection

        # 10. Build inputs
        inputs = {}
        inputs['allow_midnight_charging'] = allow_midnight_charging
        inputs['end_times_hhmm'] = end_times_hhmm
        inputs['residential_rate'] = residential_rate
        inputs['commercial_rate'] = commercial_rate
        inputs['other_rate'] = other_rate
        inputs['n_e'] = n_trips
        inputs['s_i'] = initial_soc * battery_capacity
        inputs['s_f'] = final_soc * battery_capacity
        inputs['s_ub'] = max_soc * battery_capacity
        inputs['s_lb'] = min_soc * battery_capacity
        # Penalties
        inputs['c_db'] = np.ones(n_trips) * (payment_penalty + time_penalty) * is_other
        inputs['c_dd'] = np.ones(n_trips) * dwell_charge_time_penalty
        inputs['c_ab'] = np.ones(n_trips) * (travel_penalty + payment_penalty + time_penalty)
        inputs['c_at'] = np.ones(n_trips) * travel_penalty
        inputs['c_ad'] = np.ones(n_trips) * ad_hoc_charge_time_penalty
        # Charger data
        inputs['r_d'] = destination_charger_power_array
        inputs['r_d_h'] = home_charger_power
        inputs['r_d_w'] = work_charger_power
        inputs['r_d_o'] = destination_charger_power
        inputs['r_a'] = np.ones(n_trips) * ad_hoc_charger_power
        # Time bounds for each event
        inputs['d_e_max'] = d_e_max_hours
        inputs['d_e_min'] = np.ones(n_trips) * min_dwell_event_duration
        inputs['a_e_min'] = np.ones(n_trips) * min_ad_hoc_event_duration
        inputs['a_e_max'] = np.ones(n_trips) * max_ad_hoc_event_duration
        # For the trip consumption
        inputs['d'] = trip_discharge
        inputs['b_c'] = battery_capacity
        inputs['l_i'] = trip_distances.sum()
        # location flags
        inputs['is_home'] = is_home
        inputs['is_work'] = is_work
        inputs['is_other'] = is_other
        # The final absolute start times
        inputs['absolute_start_times'] = absolute_start_hours
        # Store valid charging hours for each event
        # Cap hours to 8759 (end of the year)
        inputs['valid_hours'] = {
            e: [
                h for h in (
                    list(range((absolute_start_hours[e] // 24) * 24, ((absolute_start_hours[e] // 24) + 1) * 24))
                    if (inputs['is_home'][e] and inputs['allow_midnight_charging'][e])
                    else list(range(absolute_start_hours[e], absolute_start_hours[e] + max(1, d_e_max_hours[e]) + 1))
                )
                if 0 <= h <= 8759  # Ensure within yearly range
            ]
            for e in range(n_trips)
        }

        if isinstance(inputs['residential_rate'], list) and len(inputs['residential_rate']) == 8760:
            mean_res_rate = np.mean(inputs['residential_rate'])
            inputs['residential_rate'].append(mean_res_rate)  # Add hour 8760

        if isinstance(inputs['commercial_rate'], list) and len(inputs['commercial_rate']) == 8760:
            mean_com_rate = np.mean(inputs['commercial_rate'])
            inputs['commercial_rate'].append(mean_com_rate)  # Add hour 8760

        # If needed, transform the rate arrays from dict to list
        if isinstance(inputs['residential_rate'], dict) and "rate" in inputs['residential_rate']:
            inputs['residential_rate'] = [inputs['residential_rate']["rate"][str(hour)] for hour in range(8761)]

        if isinstance(inputs['commercial_rate'], dict) and "rate" in inputs['commercial_rate']:
            inputs['commercial_rate'] = [inputs['commercial_rate']["rate"][str(hour)] for hour in range(8761)]

        # for e in range(n_trips):
        #     print(f"Event {e}: Start Hour = {absolute_start_hours[e]}, End Hour = {absolute_start_hours[e] + d_e_max_hours[e]},dwell_time = {d_e_max_hours[e]},  Valid Hours = {inputs['valid_hours'][e]}")
        #
        # for e in range(n_trips):
        #     print(f"Event {e}: dwell times = {d_e_max_hours[e]}")
        return inputs

    def Solve(self, solver_kwargs={}):
        solver = pyomo.SolverFactory(**solver_kwargs)
        # Add MIP gap for CBC (set to 5%)
        solver.options['ratio'] = 0.20  # 5% MIP gap
        solver.options['threads'] = 20
        solver.options['maxMemory'] = 48000
        solver.options['loglevel'] = 2  # Verbose output
        solver.options['infeasibility'] = 'on'  # Show infeasibility report if supported
        # Add heuristic approach (optional)
        solver.options['heuristics'] = 'on'  # Enable solver's internal heuristic
        res = solver.solve(self.model)
        self.solver_status = res.solver.status
        self.solver_termination_condition = res.solver.termination_condition

        self.Solution()
        self.Compute


    def Build(self):

        # Pulling the keys from the inputs dict
        keys = self.inputs.keys()
        # Initializing the model as a concrete model
        # (as in one that has fixed inputted values)
        self.model = pyomo.ConcreteModel(name="EVCSP_Model")
        # Adding variables
        self.Variables()
        # Upper-level objective (inconvenience)
        self.UpperObjective()  # New
        # Lower-level KKT conditions for cost minimization
        self.LowerKKT()  # New
        # Bounds constraints
        self.Bounds()
        # Unit commitment constraints
        self.Unit_Commitment()

    def Variables(self):
        self.model.E = pyomo.Set(initialize=range(self.inputs['n_e']))

        # Destination charging
        # You can do 0..8759 or only the hours that appear in your trips.
        EH = [(e, h) for e in range(self.inputs['n_e']) for h in self.inputs['valid_hours'][e]]
        self.model.EH = pyomo.Set(initialize=EH, dimen=2)

        self.model.x_d = pyomo.Var(self.model.EH, domain=pyomo.NonNegativeReals)
        self.model.x_a = pyomo.Var(self.model.EH, domain=pyomo.NonNegativeReals)

    def UpperObjective(self):
        model = self.model

        # 1️⃣ Penalty for Events (Work & Other)
        penalty_event_work = {e: 0.1*self.inputs['c_ab'][e] for e in model.E if self.inputs['is_work'][e]}
        penalty_event_other = {e: 0.2*self.inputs['c_ab'][e] for e in model.E if self.inputs['is_other'][e]}

        # 2️⃣ Penalty for Charging (Work & Other)
        penalty_charging_work = {e: 0.1*self.inputs['c_ad'][e] for e in model.E if self.inputs['is_work'][e]}
        penalty_charging_other = {e: 0.2*self.inputs['c_ad'][e] for e in model.E if self.inputs['is_other'][e]}

        # 3️⃣ Existing Ad-hoc Penalties
        penalty_event_ad_hoc = {e: self.inputs['c_ab'][e] for e in model.E}
        penalty_charging_ad_hoc = {e: self.inputs['c_ad'][e] * self.inputs['c_at'][e] for e in model.E}

        # 4️⃣ Objective Function
        def inconvenience_expression(m):
            return sum(
                (
                    # Work penalties (only if the event is a work event)
                        penalty_event_work.get(e, 0) * m.x_d[e, h] +
                        penalty_charging_work.get(e, 0) * m.x_d[e, h] +

                        # Other location penalties (only if the event is an "other" event)
                        penalty_event_other.get(e, 0) * m.x_d[e, h] +
                        penalty_charging_other.get(e, 0) * m.x_d[e, h] +

                        # Ad-hoc penalties (always applied to x_a)
                        penalty_event_ad_hoc[e] * m.x_a[e, h] +
                        penalty_charging_ad_hoc[e] * m.x_a[e, h]
                )
                for (e, h) in m.EH
            )

        # 5️⃣ Pyomo Objective
        model.upper_objective = pyomo.Objective(
            expr=inconvenience_expression(model),
            sense=pyomo.minimize
        )

    def LowerKKT(self):
        model = self.model

        # 1) Dual variables for x_d
        model.lambda_d_min = pyomo.Var(model.EH, domain=pyomo.NonNegativeReals)
        model.lambda_d_max = pyomo.Var(model.EH, domain=pyomo.NonNegativeReals)
        model.lambda_a_min = pyomo.Var(model.EH, domain=pyomo.NonNegativeReals)
        model.lambda_a_max = pyomo.Var(model.EH, domain=pyomo.NonNegativeReals)

        # 2) Stationarity for x_d
        def stationarity_xd_rule(m, e, h):
            h_capped = min(h, 8760)  # Ensure h doesn't exceed 8760 (max hourly index for a year)

            if isinstance(self.inputs['residential_rate'], list):
                cost_rate_home = self.inputs['residential_rate'][h_capped]
            else:
                cost_rate_home = self.inputs['residential_rate']

            if isinstance(self.inputs['commercial_rate'], list):
                cost_rate_work = self.inputs['commercial_rate'][h_capped]
            else:
                cost_rate_work = self.inputs['commercial_rate']

            if isinstance(self.inputs['other_rate'], list):
                cost_rate_other = self.inputs['other_rate'][h_capped]
            else:
                cost_rate_other = self.inputs['other_rate']

            cost_rate = (
                    cost_rate_home * self.inputs['is_home'][e]
                    + cost_rate_work * self.inputs['is_work'][e]
                    + cost_rate_other * self.inputs['is_other'][e]
            )

            return cost_rate - m.lambda_d_min[e, h] + m.lambda_d_max[e, h] == 0

        model.stationarity_xd = pyomo.Constraint(model.EH, rule=stationarity_xd_rule)
        # 3) Stationarity for x_a
        def stationarity_xa_rule(m, e, h):
            if isinstance(self.inputs['other_rate'], list):
                cost_rate = self.inputs['other_rate'][h]
            else:
                cost_rate = self.inputs['other_rate']

            return cost_rate - m.lambda_a_min[e, h] + m.lambda_a_max[e, h] == 0

        model.stationarity_xa = pyomo.Constraint(model.EH, rule=stationarity_xa_rule)
        # 4) Primal feasibility: 0 <= x_d[e,h] <= r_d[e]
        def xd_min_rule(m, e, h):
            return m.x_d[e, h] >= 0


        def xd_max_rule(m, e, h):
            if self.inputs['is_home'][e]:
                max_power = min(self.inputs['r_d'][e], self.inputs['r_d_h'])
            elif self.inputs['is_work'][e]:
                max_power = min(self.inputs['r_d'][e], self.inputs['r_d_w'])
            elif self.inputs['is_other'][e]:
                max_power = self.inputs['r_d_o']
            else:
                max_power = 0  # No charging if undefined

            return m.x_d[e, h] <= max_power

        model.xd_min_con = pyomo.Constraint(model.EH, rule=xd_min_rule)
        model.xd_max_con = pyomo.Constraint(model.EH, rule=xd_max_rule)

        # Similarly for x_a: 0 <= x_a[e,h] <= r_a[e]
        def xa_min_rule(m, e, h):
            return m.x_a[e, h] >= 0

        def xa_max_rule(m, e, h):
            return m.x_a[e, h] <= self.inputs['r_a'][e]

        model.xa_min_con = pyomo.Constraint(model.EH, rule=xa_min_rule)
        model.xa_max_con = pyomo.Constraint(model.EH, rule=xa_max_rule)

    def Bounds(self):
        model = self.model
        # Create a ConstraintList to store SoC constraints
        model.bounds = pyomo.ConstraintList()
        # SoC variables for each event
        model.ess_state = pyomo.Var(model.E,
                                    domain=pyomo.NonNegativeReals,
                                    bounds=(self.inputs['s_lb'], self.inputs['s_ub']))
        model.soc_after_trip = pyomo.Var(model.E,
                                         domain=pyomo.NonNegativeReals,
                                         bounds=(self.inputs['s_lb'], self.inputs['s_ub']))
        # 1. Initial SOC at event e=0
        model.bounds.add(model.ess_state[0] >= 0.02 * self.inputs['s_i'])

        # 2. For each event e, define SoC after trip & SoC after charging
        for e in model.E:
            if e == 0:
                model.bounds.add(
                    model.soc_after_trip[e] == self.inputs['s_i'] - self.inputs['d'][e]
                )
            else:
                model.bounds.add(
                    model.soc_after_trip[e] == model.ess_state[e - 1] - self.inputs['d'][e]
                )
            # (B) SoC at the end of event e = after_trip[e] + total charged energy in event e
            model.bounds.add(
                model.ess_state[e] == model.soc_after_trip[e] + sum((model.x_d[e, h] + model.x_a[e, h]) for (e2, h) in model.EH if e2 == e)
            )

            # min_charging = 1*3600000  # Example: 2 kWh minimum charging (adjust as needed)
            # model.bounds.add(
            #     sum((model.x_d[e, h] + model.x_a[e, h]) for (e2, h) in model.EH if e2 == e) >= min_charging
            # )

        # 3. Ensure the final SOC is at least or exactly the required final SoC
        model.bounds.add(model.ess_state[self.inputs['n_e'] - 1] == self.inputs['s_f'])


    def Unit_Commitment(self):
        model = self.model

        # Pre-calculate valid hours for each event e
        valid_hours = {}
        for e in model.E:
            t_start = int(self.inputs['absolute_start_times'][e])
            t_end = int((self.inputs['absolute_start_times'][e] + self.inputs['d_e_max'][e]))
            valid_hours[e] = range(t_start, t_end + 1)

        # 1) Zero out x_d[e,h] if hour h not in the dwell window
        def no_charge_outside_window_d_rule(model, e, h):
            if h not in self.inputs['valid_hours'][e]:
                return model.x_d[e, h] == 0
            return pyomo.Constraint.Skip

        model.no_charge_outside_window_d = pyomo.Constraint(model.EH, rule=no_charge_outside_window_d_rule)

        def no_charge_outside_window_a_rule(model, e, h):
            if h not in self.inputs['valid_hours'][e]:
                return model.x_a[e, h] == 0
            return pyomo.Constraint.Skip

        model.no_charge_outside_window_a = pyomo.Constraint(model.EH, rule=no_charge_outside_window_a_rule)
        # 2) Mutual Exclusivity of Charging Modes

        def mutual_exclusivity_rule(model, e, h):
            # Ensure that x_d and x_a do not operate simultaneously beyond the charger’s capacity
            return model.x_d[e, h] + model.x_a[e, h] <= max(self.inputs['r_d'][e], self.inputs['r_a'][e])

        # Add the constraint to the model
        model.mutual_exclusivity = pyomo.Constraint(model.EH, rule=mutual_exclusivity_rule)


    def Solution(self):
        model_vars = self.model.component_map(ctype=pyomo.Var)
        serieses = []  # collection to hold the converted "serieses"
        for k in model_vars.keys():  # this is a map of {name:pyo.Var}
            v = model_vars[k]

            # make a pd.Series from each
            s = pd.Series(v.extract_values(), index=v.extract_values().keys())

            # if the series is multi-indexed we need to unstack it...
            if type(s.index[0]) == tuple:  # it is multi-indexed
                s = s.unstack(level=1)
            else:
                s = pd.DataFrame(s)  # force transition from Series -> df
            s.columns = pd.MultiIndex.from_tuples([(k, t) for t in s.columns])
            serieses.append(s)
        self.solution = pd.concat(serieses, axis=1)

    @property
    def Compute(self):

        def seconds_to_hhmm(seconds):
            hours = (seconds // 3600) % 24
            minutes = (seconds % 3600) // 60
            return int(hours * 100 + minutes)

        soc_values = np.array([pyomo.value(self.model.ess_state[e]) for e in self.model.E]) / self.inputs['b_c']
        soc_after_trip_values = np.array([pyomo.value(self.model.soc_after_trip[e]) for e in self.model.E]) / self.inputs['b_c']

        try:
            soc_values = np.array([pyomo.value(self.model.ess_state[e]) for e in self.model.E]) / self.inputs['b_c']
        except ValueError:
            print(" ess_state is uninitialized due to infeasibility. Skipping this itinerary.")
            return  # Skip further processing for this infeasible case

        # Prepend initial SOC
        soc = np.insert(soc_values, 0, self.inputs['s_i'] / self.inputs['b_c'])

        # Store SoC in the solution DataFrame
        self.solution[("soc", 0)] = soc[1:]  # SOC after charging
        self.solution[("soc_start_trip", 0)] = soc[:-1]  # SOC before the trip
        self.solution[("soc_end_trip", 0)] = soc_after_trip_values  # SOC after trip


        # 2) Compute the "inconvenience" metric (SIC)
        total_inconvenience = 0.0
        for e in self.model.E:
            penalty_e = self.inputs['c_db'][e] if self.inputs['is_other'][e] else 0.0

            # Valid charging hours for the event
            start_hour = int(self.inputs['absolute_start_times'][e])
            end_hour = int(start_hour + int(self.inputs['d_e_max'][e]))

            for h in self.inputs['valid_hours'][e]:
                if (e, h) not in self.model.EH:
                    continue
                xd_val = pyomo.value(self.model.x_d[e, h])
                total_inconvenience += penalty_e * xd_val

        self.sic = (total_inconvenience / 60) / (self.inputs['l_i'] / 1e3)

        # Extract start times and durations from itinerary

        end_times_hhmm = self.inputs.get('end_times_hhmm')
        end_times_hhmm = np.array(end_times_hhmm)
        end_times = (end_times_hhmm // 100) * 3600 + (end_times_hhmm % 100) * 60

        # 3) Calculate cost & total kWh for each event
        charging_costs = []
        charging_kwh_total = []
        charging_start_times = []
        charging_end_times = []
        hourly_charging_details = []

        for e in self.model.E:
            total_energy_j = 0.0
            total_cost_e = 0.0
            if (self.inputs['is_home'][e]
                    and self.inputs['allow_midnight_charging'][e]  # Flag for this itinerary
                    and e == max(self.model.E)):  # Last trip of the day

                # Charging starts at midnight of the same day
                charging_start_sec = (end_times[e] // 86400 + 1) * 86400  # Midnight of the next day
            else:
                # Default: Charging starts right after the trip
                charging_start_sec = end_times[e]


            # Find the last time charging occurred
            last_charging_sec = None
            first_charging_sec = None
            hourly_charging = {}

            # Calculate total charged energy and cost
            for h in self.inputs['valid_hours'][e]:
                # Check if the (e, h) pair exists in the model
                if (e, h) not in self.model.EH:
                    print(f"❌ Invalid index encountered: (e={e}, h={h})")
                    print(f"👉 EH contains {len(self.model.EH)} valid indices.")
                    print(f"Valid EH indices (first 10): {list(self.model.EH)[:10]}")
                    print(f"Valid hours for event {e}: {self.inputs['valid_hours'][e]}")
                    print(f"absolute_start_hours[{e}] = {self.inputs['absolute_start_times'][e]}")
                    print(f"d_e_max_hours[{e}] = {self.inputs['d_e_max'][e]}")
                    continue  # Skip this iteration to avoid the error

                # If valid, extract the charging values
                xd_val = pyomo.value(self.model.x_d[e, h])
                xa_val = pyomo.value(self.model.x_a[e, h])
                event_energy_h = (xd_val + xa_val) * 2.7778e-7
                total_energy_j += event_energy_h

                if event_energy_h > 1e-9:
                    if first_charging_sec is None:
                        first_charging_sec = h * 3600
                    # Use power in kW (NOT converted to kWh) to get the correct duration in seconds
                    hour_key = f"{h % 24}:00"  # Keep the full hour for clarity (e.g., 1525:00)

                    if hour_key in hourly_charging:
                        hourly_charging[hour_key] += round(event_energy_h, 3)
                    else:
                        hourly_charging[hour_key] = round(event_energy_h, 3)

                    last_charging_sec = h * 3600 + (event_energy_h / max(self.inputs['r_d'][e], self.inputs['r_a'][e]))

                h_capped = min(h, 8760)

                if self.inputs['is_home'][e]:
                    rate = self.inputs['residential_rate'][h_capped] if isinstance(self.inputs['residential_rate'], list) else self.inputs['residential_rate']
                elif self.inputs['is_work'][e]:
                    rate = self.inputs['commercial_rate'][h_capped] if isinstance(self.inputs['commercial_rate'], list) else self.inputs['commercial_rate']
                else:
                    rate = self.inputs['other_rate']

                total_cost_e += (event_energy_h * rate)

            # Convert total energy to kWh
            total_kwh = total_energy_j
            charging_kwh_total.append(total_kwh)
            charging_costs.append(total_cost_e)
            hourly_charging_details.append(hourly_charging)

            # Calculate start and end times only if there's charging
            if total_kwh > 0:
                if (self.inputs['is_home'][e] and self.inputs['allow_midnight_charging'][e] and e == max(self.model.E)):
                    # Start from midnight if allowed
                    start_sec = (end_times[e] // 86400 + 1) * 86400
                else:
                    # Default behavior
                    start_sec = charging_start_sec if first_charging_sec is None else first_charging_sec
                end_sec = last_charging_sec if last_charging_sec is not None else start_sec

                charging_start_times.append(seconds_to_hhmm(start_sec))
                charging_end_times.append(seconds_to_hhmm(end_sec))
            else:
                charging_start_times.append(np.nan)
                charging_end_times.append(np.nan)

        # Store final results
        self.solution[("charging_kwh_total", 0)] = charging_kwh_total
        self.solution[("charging_cost", 0)] = charging_costs
        self.solution[("charging_start_time_HHMM", 0)] = charging_start_times
        self.solution[("charging_end_time_HHMM", 0)] = charging_end_times
        self.solution[("hourly_charging_details", 0)] = hourly_charging_details


# class SEVCSP():
#
# 	def __init__(self,itinerary={},itinerary_kwargs={},inputs={}):
#
# 		if itinerary:
# 			self.inputs=self.ProcessItinerary(itinerary['trips'],**itinerary_kwargs)
# 		else:
# 			self.inputs=inputs
#
# 		if self.inputs:
#
# 			self.Build()
#
# 	def ProcessItinerary(self,
# 		itinerary,
# 		instances=1,
# 		home_charger_likelihood=.5,
# 		work_charger_likelihood=.1,
# 		destination_charger_likelihood=.1,
# 		consumption=478.8,
# 		battery_capacity=82*3.6e6,
# 		initial_soc=.5,
# 		final_soc=.5,
# 		ad_hoc_charger_power=121e3,
# 		home_charger_power=12.1e3,
# 		work_charger_power=12.1e3,
# 		destination_charger_power=12.1e3,
# 		ac_dc_conversion_efficiency=.88,
# 		max_soc=1,
# 		min_soc=.2,
# 		min_dwell_event_duration=15*60,
# 		max_ad_hoc_event_duration=7.2e3,
# 		min_ad_hoc_event_duration=5*60,
# 		payment_penalty=60,
# 		travel_penalty=15*60,
# 		dwell_charge_time_penalty=0,
# 		ad_hoc_charge_time_penalty=1,
# 		tiles=5,
# 		rng_seed=0,
# 		**kwargs,
# 		):
#
# 		#Generating a random seed if none provided
# 		if not rng_seed:
# 			rng_seed=np.random.randint(1e6)
#
# 		#Pulling trips for driver only
# 		# person=itinerary['PERSONID'].copy().to_numpy()
# 		# whodrove=itinerary['WHODROVE'].copy().to_numpy()
# 		# itinerary=itinerary[person==whodrove]
#
# 		#Cleaning trip and dwell durations
# 		durations=itinerary['TRVLCMIN'].copy().to_numpy()
# 		dwell_times=itinerary['DWELTIME'].copy().to_numpy()
#
# 		#Fixing any non-real dwells
# 		dwell_times[dwell_times<0]=dwell_times[dwell_times>=0].mean()
#
# 		#Padding with overnight dwell
# 		sum_of_times=durations.sum()+dwell_times[:-1].sum()
# 		if sum_of_times>=1440:
# 			ratio=1440/sum_of_times
# 			dwell_times*=ratio
# 			durations*=ratio
# 		else:
# 			final_dwell=1440-durations.sum()-dwell_times[:-1].sum()
# 			dwell_times[-1]=final_dwell
#
# 		#Trip information
# 		trip_distances=np.tile(itinerary['TRPMILES'].to_numpy(),tiles)*1609.34 #[m]
# 		trip_times=np.tile(durations,tiles)*60 #[s]
# 		trip_mean_speeds=trip_distances/trip_times #[m/s]
# 		trip_discharge=trip_distances*consumption #[J]
#
# 		#Dwell information
# 		dwells=np.tile(dwell_times,tiles)*60
# 		location_types=np.tile(itinerary['WHYTRP1S'].to_numpy(),tiles)
# 		is_home=location_types==1
# 		is_work=location_types==10
# 		is_other=((~is_home)&(~is_work))
#
# 		n_e=len(dwells)
#
# 		#Assigning chargers to dwells
# 		generator=np.random.default_rng(seed=rng_seed)
#
# 		destination_charger_power_array=np.zeros((instances,n_e))
# 		destination_charger_power_array[:,is_home]=home_charger_power
# 		destination_charger_power_array[:,is_work]=work_charger_power
# 		destination_charger_power_array[:,is_other]=destination_charger_power
#
# 		home_charger_selection=(
# 			generator.random((instances,is_home.sum()))<=home_charger_likelihood)
# 		work_charger_selection=(
# 			generator.random((instances,is_work.sum()))<=work_charger_likelihood)
# 		destination_charger_selection=(np.tile(
# 			generator.random(is_other.sum()),(instances,1))<=destination_charger_likelihood)
#
# 		destination_charger_power_array[:,is_home]*=home_charger_selection
# 		destination_charger_power_array[:,is_work]*=work_charger_selection
# 		destination_charger_power_array[:,is_other]*=destination_charger_selection
#
# 		#Assembling inputs
# 		inputs={}
#
# 		inputs['n_e']=n_e
# 		inputs['n_s']=instances
# 		inputs['s_i']=initial_soc*battery_capacity
# 		inputs['s_f']=final_soc*battery_capacity
# 		inputs['s_ub']=max_soc*battery_capacity
# 		inputs['s_lb']=min_soc*battery_capacity
# 		inputs['c_db']=np.ones(n_e)*payment_penalty*is_other
# 		inputs['c_dd']=np.ones(n_e)*dwell_charge_time_penalty
# 		inputs['c_ab']=np.ones(n_e)*(travel_penalty+payment_penalty)
# 		inputs['c_ad']=np.ones(n_e)*ad_hoc_charge_time_penalty
# 		inputs['r_d']=destination_charger_power_array
# 		inputs['r_a']=np.ones(n_e)*ad_hoc_charger_power
# 		inputs['d_e_min']=np.ones(n_e)*min_dwell_event_duration
# 		inputs['d_e_max']=dwells
# 		inputs['a_e_min']=np.ones(n_e)*min_ad_hoc_event_duration
# 		inputs['a_e_max']=np.ones(n_e)*max_ad_hoc_event_duration
# 		inputs['d']=trip_discharge
# 		inputs['b_c']=battery_capacity
# 		inputs['l_i']=trip_distances.sum()
# 		inputs['is_home']=is_home
# 		inputs['is_work']=is_work
# 		inputs['is_other']=is_other
#
# 		return inputs
#
# 	def Solve(self,solver_kwargs={}):
#
# 		solver=pyomo.SolverFactory(**solver_kwargs)
# 		res = solver.solve(self.model)
# 		self.solver_status = res.solver.status
# 		self.solver_termination_condition = res.solver.termination_condition
#
# 		self.Solution()
# 		self.Compute()
#
# 	def Build(self):
#
# 		#Pulling the keys from the inputs dict
# 		keys=self.inputs.keys()
#
# 		#Initializing the model as a concrete model
# 		#(as in one that has fixed inputted values)
# 		self.model=pyomo.ConcreteModel()
#
# 		#Adding variables
# 		self.Variables()
#
# 		#Adding the objective function
# 		self.Objective()
#
# 		#Bounds constraints
# 		self.Bounds()
#
# 		#Unit commitment constraints
# 		self.Unit_Commitment()
#
# 	def Variables(self):
#
# 		self.model.S=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_s'])])
# 		self.model.E=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_e'])])
#
# 		self.model.u_dd=pyomo.Var(self.model.S,self.model.E,domain=pyomo.NonNegativeReals)
# 		self.model.u_db=pyomo.Var(self.model.S,self.model.E,domain=pyomo.Boolean)
#
# 		self.model.u_ad=pyomo.Var(self.model.S,self.model.E,domain=pyomo.NonNegativeReals)
# 		self.model.u_ab=pyomo.Var(self.model.E,domain=pyomo.Boolean)
#
# 	def Objective(self):
#
# 		ad_hoc_charge_event=sum(
# 			self.inputs['c_ab'][e]*self.model.u_ab[e] for e in self.model.E)
#
# 		for s in self.model.S:
#
# 			destination_charge_event=sum(
# 				self.inputs['c_db'][e]*self.model.u_db[s,e] for e in self.model.E)
#
# 			if np.any(self.inputs['c_dd']>0):
# 				destination_charge_duration=sum(
# 					self.inputs['c_dd'][e]*self.model.u_dd[s,e] for e in self.model.E)
# 			else:
# 				destination_charge_duration=0
#
# 			if np.any(self.inputs['c_ad']>0):
# 				ad_hoc_charge_duration=sum(
# 					self.inputs['c_ad'][e]*self.model.u_ad[s,e] for e in self.model.E)
# 			else:
# 				ad_hoc_charge_duration=0
#
# 		self.model.objective=pyomo.Objective(expr=(
# 			destination_charge_event+
# 			destination_charge_duration+
# 			ad_hoc_charge_event+
# 			ad_hoc_charge_duration))
#
# 	def Bounds(self):
#
# 		self.model.bounds=pyomo.ConstraintList()
# 		self.model.fs=pyomo.ConstraintList()
#
# 		for s in self.model.S:
#
# 			ess_state=self.inputs['s_i']
#
# 			for e in self.model.E:
#
# 				ess_state+=(
# 					self.model.u_dd[s,e]*self.inputs['r_d'][s,e]+
# 					self.model.u_ad[s,e]*self.inputs['r_a'][e]-
# 					self.inputs['d'][e])
#
# 				self.model.bounds.add((self.inputs['s_lb'],ess_state,self.inputs['s_ub']))
#
# 			self.model.fs.add(expr=ess_state>=self.inputs['s_f'])
#
# 	def Unit_Commitment(self):
#
# 		self.model.unit_commitment=pyomo.ConstraintList()
#
# 		for s in self.model.S:
#
# 			for e in self.model.E:
#
# 				self.model.unit_commitment.add(
# 					expr=(self.inputs['d_e_min'][e]*self.model.u_db[s,e]-
# 						self.model.u_dd[s,e]<=0))
#
# 				self.model.unit_commitment.add(
# 					expr=(self.inputs['d_e_max'][e]*self.model.u_db[s,e]-
# 						self.model.u_dd[s,e]>=0))
#
# 				self.model.unit_commitment.add(
# 					expr=(self.inputs['a_e_min'][e]*self.model.u_ab[e]-
# 						self.model.u_ad[s,e]<=0))
#
# 				self.model.unit_commitment.add(
# 					expr=(self.inputs['a_e_max'][e]*self.model.u_ab[e]-
# 						self.model.u_ad[s,e]>=0))
#
# 	def Solution(self):
# 		'''
# 		From StackOverflow
# 		https://stackoverflow.com/questions/67491499/
# 		how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
# 		'''
# 		model_vars=self.model.component_map(ctype=pyomo.Var)
#
# 		serieses=[]   # collection to hold the converted "serieses"
# 		for k in model_vars.keys():   # this is a map of {name:pyo.Var}
# 			v=model_vars[k]
#
# 			# make a pd.Series from each
# 			s=pd.Series(v.extract_values(),index=v.extract_values().keys())
#
# 			# if the series is multi-indexed we need to unstack it...
# 			if type(s.index[0])==tuple:# it is multi-indexed
# 				s=s.unstack(level=0)
# 			else:
# 				s=pd.DataFrame(s) # force transition from Series -> df
#
# 			s.columns=pd.MultiIndex.from_tuples([(k,t) for t in s.columns])
#
# 			serieses.append(s)
#
# 		self.solution=pd.concat(serieses,axis=1)
#
# 	def Compute(self):
# 		'''
# 		Computing SOC in time and SIC for itinerary
# 		'''
#
# 		sic=np.zeros(self.inputs['n_s'])
#
# 		for s in range(self.inputs['n_s']):
#
# 			ess_state=np.append(self.inputs['s_i'],
# 				self.inputs['s_i']+np.cumsum(-self.inputs['d'])+
# 				np.cumsum(self.solution['u_dd'][s]*self.inputs['r_d'][s])+
# 				np.cumsum(self.solution['u_ad'][s]*self.inputs['r_a'][s])
# 				)
# 			soc=ess_state/self.inputs['b_c']
#
# 			sic[s]=((
# 				sum(self.solution['u_db'][s]*self.inputs['c_db'])+
# 				sum(self.solution['u_dd'][s]*self.inputs['c_dd'])+
# 				sum(self.solution['u_ab'][0]*self.inputs['c_ab'])+
# 				sum(self.solution['u_ad'][s]*self.inputs['c_ad'])
# 				)/60)/(self.inputs['l_i']/1e3)
#
# 			self.solution['soc',s]=soc[1:]
#
# 		self.sic=sic


def SeparateInputs(inputs):
    n_s_list = [val.shape for key, val in inputs.items() if hasattr(val, 'shape')]
    n_s_list = [val[0] for val in n_s_list if len(val) > 1]
    n_s = max(n_s_list)

    separated_inputs_list = [None] * n_s

    for idx in range(n_s):

        separated_inputs = {}

        for key, val in inputs.items():

            if hasattr(val, 'shape'):
                if len(val.shape) > 1:
                    separated_inputs[key] = val[idx]
                else:
                    separated_inputs[key] = val
            else:
                separated_inputs[key] = val

        separated_inputs_list[idx] = separated_inputs

    return separated_inputs_list


def RunDeterministic(vehicle_class, itinerary, itinerary_kwargs={}, solver_kwargs={}):
    problem = vehicle_class(itinerary, itinerary_kwargs=itinerary_kwargs)
    problem.Solve(solver_kwargs)

    return problem.sic, problem


def RunStochastic(itinerary, itinerary_kwargs={}, solver_kwargs={}):
    problem = SEVCSP(itinerary, itinerary_kwargs=itinerary_kwargs)
    problem.Solve(solver_kwargs)

    return problem.sic, problem


def RunAsIndividuals(problem, solver_kwargs={}):
    separated_inputs_list = SeparateInputs(problem.inputs)

    problems = [None] * len(separated_inputs_list)
    sic = np.zeros(len(separated_inputs_list))

    for idx in range(len(separated_inputs_list)):
        problem = EVCSP(inputs=separated_inputs_list[idx])
        problem.Solve(solver_kwargs)

        problems[idx] = problem
        sic[idx] = problem.sic

    return sic, problems
