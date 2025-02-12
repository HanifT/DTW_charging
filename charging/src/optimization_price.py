import numpy as np
import pandas as pd
import pyomo.environ as pyomo


class EVCSP():

	def __init__(self, itinerary={}, itinerary_kwargs={}, inputs={}):

		if itinerary:
			self.inputs=self.ProcessItinerary(itinerary['trips'], **itinerary_kwargs)
		else:
			self.inputs=inputs

		if self.inputs:

			self.Build()

	def ProcessItinerary(self,
		itinerary,
		home_charger_likelihood=1,
		work_charger_likelihood=0.75,
		destination_charger_likelihood=.1,
		consumption=782.928,  # J/meter
		battery_capacity=60*3.6e6,  # J
		initial_soc=.5,
		final_soc=.5,
		ad_hoc_charger_power=100e3,
		home_charger_power=6.1e3,
		work_charger_power=6.1e3,
		destination_charger_power=100.1e3,
		ac_dc_conversion_efficiency=.88,
		max_soc=1,
		min_soc=.2,
		min_dwell_event_duration=15*60,
		max_ad_hoc_event_duration=7.2e3,
		min_ad_hoc_event_duration=5*60,
		payment_penalty=60,
		travel_penalty=1*60,
		dwell_charge_time_penalty=0,
		ad_hoc_charge_time_penalty=1,
		tiles=7,
		rng_seed=0,
		**kwargs,
		):

		#Generating a random seed if none provided
		if not rng_seed:
			rng_seed=np.random.randint(1e6)

		#Cleaning trip and dwell durations
		durations=itinerary['TRVLCMIN'].copy().to_numpy()
		dwell_times=itinerary['DWELTIME'].copy().to_numpy()
		# Replace negative values with zero
		dwell_times = np.where(dwell_times < 0, 0, dwell_times)

		#Fixing any non-real dwells
		dwell_times[dwell_times<0]=dwell_times[dwell_times>=0].mean()
		
		#Padding with overnight dwell
		sum_of_times=durations.sum()+dwell_times[:-1].sum()

		# Convert to float before applying ratio to allow float multiplication
		dwell_times = dwell_times.astype(float)
		durations = durations.astype(float)

		# Adjust dwell_times and durations based on the calculated ratio
		if sum_of_times >= 1440:
			ratio = 1440 / sum_of_times
			dwell_times *= ratio
			durations *= ratio
		else:
			final_dwell = 1440 - durations.sum() - dwell_times[:-1].sum()
			dwell_times[-1] = final_dwell

		#Trip information
		trip_distances=np.tile(itinerary['TRPMILES'].to_numpy(),tiles)*1609.34 #[m]
		trip_times=np.tile(durations,tiles)*60 #[s]
		trip_mean_speeds=trip_distances/trip_times #[m/s]
		trip_discharge=trip_distances*consumption #[J]

		#Dwell information
		dwells=np.tile(dwell_times,tiles)*60
		location_types=np.tile(itinerary['WHYTRP1S'].to_numpy(),tiles)
		is_home=location_types==1
		is_work=location_types==10
		is_other=((~is_home)&(~is_work))

		n_e=len(dwells)

		#Assigning chargers to dwells
		generator=np.random.default_rng(seed=rng_seed)

		destination_charger_power_array=np.zeros(n_e)
		destination_charger_power_array[is_home]=home_charger_power
		destination_charger_power_array[is_work]=work_charger_power
		destination_charger_power_array[is_other]=destination_charger_power

		home_charger_selection=(
			generator.random(is_home.sum())<=home_charger_likelihood)
		work_charger_selection=(
			generator.random(is_work.sum())<=work_charger_likelihood)
		destination_charger_selection=(
			generator.random(is_other.sum())<=destination_charger_likelihood)

		destination_charger_power_array[is_home]*=home_charger_selection
		destination_charger_power_array[is_work]*=work_charger_selection
		destination_charger_power_array[is_other]*=destination_charger_selection

		#Assembling inputs
		inputs={}

		inputs['n_e']=n_e
		inputs['s_i']=initial_soc*battery_capacity
		inputs['s_f']=final_soc*battery_capacity
		inputs['s_ub']=max_soc*battery_capacity
		inputs['s_lb']=min_soc*battery_capacity
		inputs['c_db']=np.ones(n_e)*payment_penalty*is_other
		inputs['c_dd']=np.ones(n_e)*dwell_charge_time_penalty
		inputs['c_ab']=np.ones(n_e)*(travel_penalty+payment_penalty)
		inputs['c_ad']=np.ones(n_e)*ad_hoc_charge_time_penalty
		inputs['r_d']=destination_charger_power_array
		inputs['r_a']=np.ones(n_e)*ad_hoc_charger_power
		inputs['d_e_min']=np.ones(n_e)*min_dwell_event_duration
		inputs['d_e_max']=dwells
		inputs['a_e_min']=np.ones(n_e)*min_ad_hoc_event_duration
		inputs['a_e_max']=np.ones(n_e)*max_ad_hoc_event_duration
		inputs['d']=trip_discharge
		inputs['b_c']=battery_capacity
		inputs['l_i']=trip_distances.sum()
		inputs['is_home']=is_home
		inputs['is_work']=is_work
		inputs['is_other']=is_other

		return inputs

	def Solve(self,solver_kwargs={}):

		solver=pyomo.SolverFactory(**solver_kwargs)
		res = solver.solve(self.model)
		self.solver_status = res.solver.status
		self.solver_termination_condition = res.solver.termination_condition

		self.Solution()
		self.Compute()

	def Build(self):

		#Pulling the keys from the inputs dict
		keys=self.inputs.keys()

		#Initializing the model as a concrete model
		#(as in one that has fixed inputted values)
		self.model=pyomo.ConcreteModel()

		#Adding variables
		self.Variables()

		#Adding the objective function
		self.Objective()

		#Bounds constraints
		self.Bounds()

		#Unit commitment constraints
		self.Unit_Commitment()

	def Variables(self):

		self.model.E=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_e'])])

		self.model.u_dd=pyomo.Var(self.model.E,domain=pyomo.NonNegativeReals)
		self.model.u_db=pyomo.Var(self.model.E,domain=pyomo.Boolean)

		self.model.u_ad=pyomo.Var(self.model.E,domain=pyomo.NonNegativeReals)
		self.model.u_ab=pyomo.Var(self.model.E,domain=pyomo.Boolean)

	def Objective(self):

		destination_charge_event=sum(
			self.inputs['c_db'][e]*self.model.u_db[e] for e in self.model.E)

		if np.any(self.inputs['c_dd']>0):
			destination_charge_duration=sum(
				self.inputs['c_dd'][e]*self.model.u_dd[e] for e in self.model.E)
		else:
			destination_charge_duration=0

		ad_hoc_charge_event=sum(
			self.inputs['c_ab'][e]*self.model.u_ab[e] for e in self.model.E)

		if np.any(self.inputs['c_ad']>0):
			ad_hoc_charge_duration=sum(
				self.inputs['c_ad'][e]*self.model.u_ad[e] for e in self.model.E)
		else:
			ad_hoc_charge_duration=0

		self.model.objective=pyomo.Objective(expr=(
			destination_charge_event+
			destination_charge_duration+
			ad_hoc_charge_event+
			ad_hoc_charge_duration))

	def Bounds(self):

		self.model.bounds=pyomo.ConstraintList()
		self.model.fs=pyomo.ConstraintList()

		ess_state=self.inputs['s_i']

		for e in range(self.inputs['n_e']):

			ess_state+=(
				self.model.u_dd[e]*self.inputs['r_d'][e]+
				self.model.u_ad[e]*self.inputs['r_a'][e]-
				self.inputs['d'][e])

			self.model.bounds.add((self.inputs['s_lb'],ess_state,self.inputs['s_ub']))

		self.model.fs.add(expr=ess_state==self.inputs['s_f'])

	def Unit_Commitment(self):

		self.model.unit_commitment=pyomo.ConstraintList()

		for e in range(self.inputs['n_e']):

			self.model.unit_commitment.add(
				expr=(self.inputs['d_e_min'][e]*self.model.u_db[e]-self.model.u_dd[e]<=0))
			
			self.model.unit_commitment.add(
				expr=(self.inputs['d_e_max'][e]*self.model.u_db[e]-self.model.u_dd[e]>=0))

			self.model.unit_commitment.add(
				expr=(self.inputs['a_e_min'][e]*self.model.u_ab[e]-self.model.u_ad[e]<=0))
			
			self.model.unit_commitment.add(
				expr=(self.inputs['a_e_max'][e]*self.model.u_ab[e]-self.model.u_ad[e]>=0))

	def Solution(self):
		'''
		From StackOverflow
		https://stackoverflow.com/questions/67491499/
		how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
		'''
		model_vars=self.model.component_map(ctype=pyomo.Var)

		serieses=[]   # collection to hold the converted "serieses"
		for k in model_vars.keys():   # this is a map of {name:pyo.Var}
			v=model_vars[k]

			# make a pd.Series from each    
			s=pd.Series(v.extract_values(),index=v.extract_values().keys())

			# if the series is multi-indexed we need to unstack it...
			if type(s.index[0])==tuple:# it is multi-indexed
				s=s.unstack(level=1)
			else:
				s=pd.DataFrame(s) # force transition from Series -> df

			s.columns=pd.MultiIndex.from_tuples([(k,t) for t in s.columns])

			serieses.append(s)

		self.solution=pd.concat(serieses,axis=1)

	def Compute(self):
		'''
		Computing SOC in time and SIC for itinerary
		'''

		ess_state=np.append(self.inputs['s_i'],
			self.inputs['s_i']+np.cumsum(-self.inputs['d'])+
			np.cumsum(self.solution['u_dd'][0]*self.inputs['r_d'])+
			np.cumsum(self.solution['u_ad'][0]*self.inputs['r_a'])
			)
		soc=ess_state/self.inputs['b_c']

		self.sic=((
			sum(self.solution['u_db'][0]*self.inputs['c_db'])+
			sum(self.solution['u_dd'][0]*self.inputs['c_dd'])*+
			sum(self.solution['u_ab'][0]*self.inputs['c_ab'])+
			sum(self.solution['u_ad'][0]*self.inputs['c_ad'])
			)/60)/(self.inputs['l_i']/1e3)

		self.solution['soc',0]=soc[1:]

class SEVCSP():

	def __init__(self,itinerary={},itinerary_kwargs={},inputs={}):

		if itinerary:
			self.inputs=self.ProcessItinerary(itinerary['trips'],**itinerary_kwargs)
		else:
			self.inputs=inputs

		if self.inputs:

			self.Build()

	def ProcessItinerary(self,
		itinerary,
		instances=1,
		home_charger_likelihood=.5,
		work_charger_likelihood=.1,
		destination_charger_likelihood=.1,
		consumption=478.8,
		battery_capacity=82*3.6e6,
		initial_soc=.5,
		final_soc=.5,
		ad_hoc_charger_power=121e3,
		home_charger_power=12.1e3,
		work_charger_power=12.1e3,
		destination_charger_power=12.1e3,
		ac_dc_conversion_efficiency=.88,
		max_soc=1,
		min_soc=.2,
		min_dwell_event_duration=15*60,
		max_ad_hoc_event_duration=7.2e3,
		min_ad_hoc_event_duration=5*60,
		payment_penalty=60,
		travel_penalty=15*60,
		dwell_charge_time_penalty=0,
		ad_hoc_charge_time_penalty=1,
		tiles=5,
		rng_seed=0,
		**kwargs,
		):

		#Generating a random seed if none provided
		if not rng_seed:
			rng_seed=np.random.randint(1e6)

		#Pulling trips for driver only
		# person=itinerary['PERSONID'].copy().to_numpy()
		# whodrove=itinerary['WHODROVE'].copy().to_numpy()
		# itinerary=itinerary[person==whodrove]

		#Cleaning trip and dwell durations
		durations=itinerary['TRVLCMIN'].copy().to_numpy()
		dwell_times=itinerary['DWELTIME'].copy().to_numpy()

		#Fixing any non-real dwells
		dwell_times[dwell_times<0]=dwell_times[dwell_times>=0].mean()

		#Padding with overnight dwell
		sum_of_times=durations.sum()+dwell_times[:-1].sum()
		if sum_of_times>=1440:
			ratio=1440/sum_of_times
			dwell_times*=ratio
			durations*=ratio
		else:
			final_dwell=1440-durations.sum()-dwell_times[:-1].sum()
			dwell_times[-1]=final_dwell

		#Trip information
		trip_distances=np.tile(itinerary['TRPMILES'].to_numpy(),tiles)*1609.34 #[m]
		trip_times=np.tile(durations,tiles)*60 #[s]
		trip_mean_speeds=trip_distances/trip_times #[m/s]
		trip_discharge=trip_distances*consumption #[J]

		#Dwell information
		dwells=np.tile(dwell_times,tiles)*60
		location_types=np.tile(itinerary['WHYTRP1S'].to_numpy(),tiles)
		is_home=location_types==1
		is_work=location_types==10
		is_other=((~is_home)&(~is_work))

		n_e=len(dwells)

		#Assigning chargers to dwells
		generator=np.random.default_rng(seed=rng_seed)

		destination_charger_power_array=np.zeros((instances,n_e))
		destination_charger_power_array[:,is_home]=home_charger_power
		destination_charger_power_array[:,is_work]=work_charger_power
		destination_charger_power_array[:,is_other]=destination_charger_power

		home_charger_selection=(
			generator.random((instances,is_home.sum()))<=home_charger_likelihood)
		work_charger_selection=(
			generator.random((instances,is_work.sum()))<=work_charger_likelihood)
		# print(generator.random((instances,is_work.sum()))<=work_charger_likelihood)
		# print(work_charger_likelihood)
		destination_charger_selection=(np.tile(
			generator.random(is_other.sum()),(instances,1))<=destination_charger_likelihood)

		destination_charger_power_array[:,is_home]*=home_charger_selection
		destination_charger_power_array[:,is_work]*=work_charger_selection
		destination_charger_power_array[:,is_other]*=destination_charger_selection

		#Assembling inputs
		inputs={}

		inputs['n_e']=n_e
		inputs['n_s']=instances
		inputs['s_i']=initial_soc*battery_capacity
		inputs['s_f']=final_soc*battery_capacity
		inputs['s_ub']=max_soc*battery_capacity
		inputs['s_lb']=min_soc*battery_capacity
		inputs['c_db']=np.ones(n_e)*payment_penalty*is_other
		inputs['c_dd']=np.ones(n_e)*dwell_charge_time_penalty
		inputs['c_ab']=np.ones(n_e)*(travel_penalty+payment_penalty)
		inputs['c_ad']=np.ones(n_e)*ad_hoc_charge_time_penalty
		inputs['r_d']=destination_charger_power_array
		inputs['r_a']=np.ones(n_e)*ad_hoc_charger_power
		inputs['d_e_min']=np.ones(n_e)*min_dwell_event_duration
		inputs['d_e_max']=dwells
		inputs['a_e_min']=np.ones(n_e)*min_ad_hoc_event_duration
		inputs['a_e_max']=np.ones(n_e)*max_ad_hoc_event_duration
		inputs['d']=trip_discharge
		inputs['b_c']=battery_capacity
		inputs['l_i']=trip_distances.sum()
		inputs['is_home']=is_home
		inputs['is_work']=is_work
		inputs['is_other']=is_other

		return inputs

	def Solve(self,solver_kwargs={}):

		solver=pyomo.SolverFactory(**solver_kwargs)
		res = solver.solve(self.model)
		self.solver_status = res.solver.status
		self.solver_termination_condition = res.solver.termination_condition

		self.Solution()
		self.Compute()

	def Build(self):

		#Pulling the keys from the inputs dict
		keys=self.inputs.keys()

		#Initializing the model as a concrete model
		#(as in one that has fixed inputted values)
		self.model=pyomo.ConcreteModel()

		#Adding variables
		self.Variables()

		#Adding the objective function
		self.Objective()

		#Bounds constraints
		self.Bounds()

		#Unit commitment constraints
		self.Unit_Commitment()

	def Variables(self):

		self.model.S=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_s'])])
		self.model.E=pyomo.Set(initialize=[idx for idx in range(self.inputs['n_e'])])

		self.model.u_dd=pyomo.Var(self.model.S,self.model.E,domain=pyomo.NonNegativeReals)
		self.model.u_db=pyomo.Var(self.model.S,self.model.E,domain=pyomo.Boolean)

		self.model.u_ad=pyomo.Var(self.model.S,self.model.E,domain=pyomo.NonNegativeReals)
		self.model.u_ab=pyomo.Var(self.model.E,domain=pyomo.Boolean)

	def Objective(self):

		ad_hoc_charge_event=sum(
			self.inputs['c_ab'][e]*self.model.u_ab[e] for e in self.model.E)

		for s in self.model.S:

			destination_charge_event=sum(
				self.inputs['c_db'][e]*self.model.u_db[s,e] for e in self.model.E)

			if np.any(self.inputs['c_dd']>0):
				destination_charge_duration=sum(
					self.inputs['c_dd'][e]*self.model.u_dd[s,e] for e in self.model.E)
			else:
				destination_charge_duration=0

			if np.any(self.inputs['c_ad']>0):
				ad_hoc_charge_duration=sum(
					self.inputs['c_ad'][e]*self.model.u_ad[s,e] for e in self.model.E)
			else:
				ad_hoc_charge_duration=0

		self.model.objective=pyomo.Objective(expr=(
			destination_charge_event+
			destination_charge_duration+
			ad_hoc_charge_event+
			ad_hoc_charge_duration))

	def Bounds(self):

		self.model.bounds=pyomo.ConstraintList()
		self.model.fs=pyomo.ConstraintList()

		for s in self.model.S:

			ess_state=self.inputs['s_i']

			for e in self.model.E:

				ess_state+=(
					self.model.u_dd[s,e]*self.inputs['r_d'][s,e]+
					self.model.u_ad[s,e]*self.inputs['r_a'][e]-
					self.inputs['d'][e])

				self.model.bounds.add((self.inputs['s_lb'],ess_state,self.inputs['s_ub']))

			self.model.fs.add(expr=ess_state>=self.inputs['s_f'])

	def Unit_Commitment(self):

		self.model.unit_commitment=pyomo.ConstraintList()

		for s in self.model.S:

			for e in self.model.E:

				self.model.unit_commitment.add(
					expr=(self.inputs['d_e_min'][e]*self.model.u_db[s,e]-
						self.model.u_dd[s,e]<=0))

				self.model.unit_commitment.add(
					expr=(self.inputs['d_e_max'][e]*self.model.u_db[s,e]-
						self.model.u_dd[s,e]>=0))

				self.model.unit_commitment.add(
					expr=(self.inputs['a_e_min'][e]*self.model.u_ab[e]-
						self.model.u_ad[s,e]<=0))

				self.model.unit_commitment.add(
					expr=(self.inputs['a_e_max'][e]*self.model.u_ab[e]-
						self.model.u_ad[s,e]>=0))

	def Solution(self):
		'''
		From StackOverflow
		https://stackoverflow.com/questions/67491499/
		how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
		'''
		model_vars=self.model.component_map(ctype=pyomo.Var)

		serieses=[]   # collection to hold the converted "serieses"
		for k in model_vars.keys():   # this is a map of {name:pyo.Var}
			v=model_vars[k]

			# make a pd.Series from each
			s=pd.Series(v.extract_values(),index=v.extract_values().keys())

			# if the series is multi-indexed we need to unstack it...
			if type(s.index[0])==tuple:# it is multi-indexed
				s=s.unstack(level=0)
			else:
				s=pd.DataFrame(s) # force transition from Series -> df

			s.columns=pd.MultiIndex.from_tuples([(k,t) for t in s.columns])

			serieses.append(s)

		self.solution=pd.concat(serieses,axis=1)

	def Compute(self):
		'''
		Computing SOC in time and SIC for itinerary
		'''

		sic=np.zeros(self.inputs['n_s'])

		for s in range(self.inputs['n_s']):

			ess_state=np.append(self.inputs['s_i'],
				self.inputs['s_i']+np.cumsum(-self.inputs['d'])+
				np.cumsum(self.solution['u_dd'][s]*self.inputs['r_d'][s])+
				np.cumsum(self.solution['u_ad'][s]*self.inputs['r_a'][s])
				)
			soc=ess_state/self.inputs['b_c']

			sic[s]=((
				sum(self.solution['u_db'][s]*self.inputs['c_db'])+
				sum(self.solution['u_dd'][s]*self.inputs['c_dd'])+
				sum(self.solution['u_ab'][0]*self.inputs['c_ab'])+
				sum(self.solution['u_ad'][s]*self.inputs['c_ad'])
				)/60)/(self.inputs['l_i']/1e3)

			self.solution['soc',s]=soc[1:]

		self.sic=sic


def SeparateInputs(inputs):

	n_s_list=[val.shape for key,val in inputs.items() if hasattr(val,'shape')]
	n_s_list=[val[0] for val in n_s_list if len(val)>1]
	n_s=max(n_s_list)

	separated_inputs_list=[None]*n_s

	for idx in range(n_s):

		separated_inputs={}

		for key,val in inputs.items():

			if hasattr(val,'shape'):
				if len(val.shape)>1:
					separated_inputs[key]=val[idx]
				else:
					separated_inputs[key]=val
			else:
				separated_inputs[key]=val

		separated_inputs_list[idx]=separated_inputs

	return separated_inputs_list

def RunDeterministic(vehicle_class,itinerary,itinerary_kwargs={},solver_kwargs={}):

	problem=vehicle_class(itinerary,itinerary_kwargs=itinerary_kwargs)
	problem.Solve(solver_kwargs)

	return problem.sic,problem

def RunStochastic(itinerary,itinerary_kwargs={},solver_kwargs={}):

	problem=SEVCSP(itinerary,itinerary_kwargs=itinerary_kwargs)
	problem.Solve(solver_kwargs)

	return problem.sic,problem

def RunAsIndividuals(problem,solver_kwargs={}):

	separated_inputs_list=SeparateInputs(problem.inputs)
	
	problems=[None]*len(separated_inputs_list)
	sic=np.zeros(len(separated_inputs_list))

	for idx in range(len(separated_inputs_list)):

		problem=EVCSP(inputs=separated_inputs_list[idx])
		problem.Solve(solver_kwargs)
		
		problems[idx]=problem
		sic[idx]=problem.sic

	return sic,problems