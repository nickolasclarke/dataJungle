%% Script Description Header

% File Name: OSESMO_Option_S.m
% File Location: "~/Desktop/OSESMO Git Repository"
% Project: Open-Source Energy Storage Model (OSESMO)
% Description: Simulates operation of energy storage system.
% Calculates customer savings, GHG reduction, and battery cycling.

function OSESMO_Option_S(Modeling_Team_Input, Model_Run_Number_Input, Model_Type_Input, ...
    Model_Timestep_Resolution, Customer_Class_Input, Load_Profile_Name_Input, ...
    Retail_Rate_Name_Input, Solar_Profile_Name_Input, Solar_Size_Input, ...
    Storage_Type_Input, Storage_Power_Rating_Input, Usable_Storage_Capacity_Input, ...
    Single_Cycle_RTE_Input, Parasitic_Storage_Load_Input, ...
    Storage_Control_Algorithm_Name, GHG_Reduction_Solution_Input, Equivalent_Cycling_Constraint_Input, ...
    Annual_RTE_Constraint_Input, ITC_Constraint_Input, ...
    Carbon_Adder_Incentive_Value_Input, Emissions_Forecast_Signal_Input, ...
    OSESMO_Git_Repo_Directory, Input_Output_Data_Directory_Location, Start_Time_Input, ...
    Show_Plots, Export_Plots, Export_Data, ...
    Solar_Installed_Cost_per_kW, Storage_Installed_Cost_per_kWh, Estimated_Future_Lithium_Ion_Battery_Installed_Cost_per_kWh, ...
    Cycle_Life, Storage_Depth_of_Discharge, Initial_Final_SOC, End_of_Month_Padding_Days)

%% Calculate Model Variable Values from User-Specified Input Values

% Convert model timestep resolution input from minutes to hours.
% This is a more useful format for the model to use.
delta_t = (Model_Timestep_Resolution/60); % Model timestep resolution, in hours.

% Convert storage efficiency from round-trip efficiency to charge and discharge efficiency.
% Charge efficiency and discharge efficiency assumed to be square root of round-trip efficiency (Eff_c = Eff_d).
% Round-trip efficiency taken from Lazard's Levelized Cost of Storage report (2017), pg. 130
% https://www.lazard.com/media/450338/lazard-levelized-cost-of-storage-version-30.pdf
Eff_c = sqrt(Single_Cycle_RTE_Input);
Eff_d = sqrt(Single_Cycle_RTE_Input);

% Parasitic storage load (kW) calculated based on input value, which is
% given as a percentage of Storage Power Rating.
Parasitic_Storage_Load = Storage_Power_Rating_Input * Parasitic_Storage_Load_Input;

% Set Carbon Adder to $0/metric ton if GHG Reduction Solution is not GHG Signal Co-Optimization.
% This serves as error-handling in case the user sets the Carbon Adder to a
% non-zero value, and sets the GHG Reduction Solution to something other
% than GHG Signal Co-Optimization.

if GHG_Reduction_Solution_Input ~= "GHG Signal Co-Optimization"
    Carbon_Adder_Incentive_Value_Input = 0; % Value of carbon adder, in $ per metric ton.  
    Emissions_Forecast_Signal_Input = "No Emissions Forecast Signal"; % Ensures consistent outputs.
end

% Set Solar Profile Name Input to "No Solar", set Solar Size Input to 0 kW,
% and set ITC Constraint to 0 if Model Type Input is Storage Only.
% This serves as error handling.

if Model_Type_Input == "Storage Only"
    Solar_Profile_Name_Input = "No Solar";
    Solar_Size_Input = 0;
    ITC_Constraint_Input = 0;
end

% Throw an error if Model Type Input is set to Solar Plus Storage
% and Solar Profile Name Input is set to "No Solar",
% or if Solar Size Input is set to 0 kW.

if Model_Type_Input == "Solar Plus Storage"  
    if Solar_Profile_Name_Input == "No Solar"
        error("Solar Plus Storage Model selected, but No Solar Profile Name Input selected.")
    end
    
    if Solar_Size_Input == 0
        error("Solar Plus Storage Model selected, but Solar Size Input set to 0 kW.")
    end
end

% Throw an error if Storage Control Algorithm set to OSESMO Non-Economic
% Solar Self-Supply, and Model Type Input is set to Storage Only,
% or if Solar Profile Name Input is set to "No Solar",
% or if Solar Size Input is set to 0 kW.

if Storage_Control_Algorithm_Name == "OSESMO Non-Economic Solar Self-Supply"
    if Model_Type_Input == "Storage Only"
        error("OSESMO Non-Economic Solar Self-Supply control algorithm selected, but Model Type set to Storage Only.")
    end
    
    if Solar_Profile_Name_Input == "No Solar"
        error("OSESMO Non-Economic Solar Self-Supply control algorithm selected, but No Solar Profile Name Input selected.")
    end
    
    if Solar_Size_Input == 0
        error("OSESMO Non-Economic Solar Self-Supply control algorithm selected, but Solar Size Input set to 0 kW.")
    end  
end


% Emissions Evaluation Signal
% Real-time five-minute marginal emissions signal used to evaluate emission impacts.
% Available for both NP15 (Northern California congestion zone) and SP15
% (Southern California congestion zone) (using Itron/E3 implied heat-rate methodology.
% Mapped to model runs based on retail rate.
% Note: For OSESMO model runs detailed in Working Group Report,
% emissions-rate evaluation signals were mapped to load profiles.

if contains(Retail_Rate_Name_Input, "PG&E")
    Emissions_Evaluation_Signal_Input = "NP15 RT5M";
    
elseif contains(Retail_Rate_Name_Input, "SCE") || contains(Retail_Rate_Name_Input, "SDG&E")
    Emissions_Evaluation_Signal_Input = "SP15 RT5M";    
end


% Total Storage Capacity
% Total storage capacity is the total chemical capacity of the battery.
% The usable storage capacity is equal to the total storage capacity 
% multiplied by storage depth of discharge. This means that the total
% storage capacity is equal to the usable storage capacity divided by
% storage depth of discharge. Total storage capacity is used to 
% calculate battery cost, whereas usable battery capacity is used 
% as an input to operational simulation portion of model.
Total_Storage_Capacity = Usable_Storage_Capacity_Input/Storage_Depth_of_Discharge;

% Usable Storage Capacity
% Usable storage capacity is equal to the original usable storage capacity
% input, degraded every month based on the number of cycles performed in
% that month. Initialized at the usable storage capacity input value.

Usable_Storage_Capacity = Usable_Storage_Capacity_Input;


% Cycling Penalty
% Cycling penalty for lithium-ion battery is equal to estimated replacement cell cost
% in 10 years divided by expected cycle life. Cycling penalty for flow batteries is $0/cycle.

if Storage_Type_Input == "Lithium-Ion Battery"
    cycle_pen = (Total_Storage_Capacity * Estimated_Future_Lithium_Ion_Battery_Installed_Cost_per_kWh) / Cycle_Life;   
elseif Storage_Type_Input == "Flow Battery"
    cycle_pen = 0;  
end


%% Import Data from CSV Files

% Begin script runtime timer
tstart = tic;

% Import Load Profile Data
% Call Import_Load_Profile_Data function.
[Load_Profile_Data, Load_Profile_Master_Index] = Import_Load_Profile_Data(Input_Output_Data_Directory_Location, OSESMO_Git_Repo_Directory, delta_t, Load_Profile_Name_Input);

Annual_Peak_Demand_Baseline = max(Load_Profile_Data);
Annual_Total_Energy_Consumption_Baseline = sum(Load_Profile_Data) * delta_t;

% Import Marginal Emissions Rate Data Used as Forecast
% Call Import_Marginal_Emissions_Rate_Forecast_Data function.
Marginal_Emissions_Rate_Forecast_Data = Import_Marginal_Emissions_Rate_Forecast_Data(Input_Output_Data_Directory_Location, OSESMO_Git_Repo_Directory, ...
    delta_t, Load_Profile_Data, Emissions_Forecast_Signal_Input);


% Import Marginal Emissions Rate Data Used for Evaluation
% Call Import_Marginal_Emissions_Rate_Forecast_Data function.
Marginal_Emissions_Rate_Evaluation_Data = Import_Marginal_Emissions_Rate_Evaluation_Data(Input_Output_Data_Directory_Location, OSESMO_Git_Repo_Directory, ...
    delta_t, Emissions_Evaluation_Signal_Input);


% Import Carbon Adder Data
% Carbon Adder ($/kWh) = Marginal Emissions Rate (metric tons CO2/MWh) * ...
% Carbon Adder ($/metric ton) * (1 MWh/1000 kWh)
Carbon_Adder_Data = (Marginal_Emissions_Rate_Forecast_Data * ...
    Carbon_Adder_Incentive_Value_Input)/1000;


% Import Retail Rate Data
% Call Import_Retail_Rate_Data function.
[Retail_Rate_Master_Index, Retail_Rate_Effective_Date, ...
    Volumetric_Rate_Data, Summer_Peak_DC, Summer_Peak_DC_Period, ...
    Summer_Part_Peak_DC, Summer_Part_Peak_DC_Period, ...
    Summer_Special_Maximum_DC, Summer_Special_Maximum_DC_Period, ...
    Summer_Noncoincident_DC, Summer_Noncoincident_DC_Period, ...
    Winter_Peak_DC, Winter_Peak_DC_Period, ...
    Winter_Part_Peak_DC, Winter_Part_Peak_DC_Period, ...
    Winter_Special_Maximum_DC, Winter_Special_Maximum_DC_Period, ...
    Winter_Noncoincident_DC, Winter_Noncoincident_DC_Period, ...
    Fixed_Per_Meter_Day_Charge, Fixed_Per_Meter_Month_Charge, ...
    First_Summer_Month, Last_Summer_Month, Month_Data, Day_Data, ...
    Summer_Peak_Binary_Data, Summer_Part_Peak_Binary_Data, ...
    Winter_Peak_Binary_Data, Winter_Part_Peak_Binary_Data, Special_Maximum_Demand_Binary_Data] = Import_Option_S_Retail_Rate_Data(Input_Output_Data_Directory_Location, OSESMO_Git_Repo_Directory, ...
    delta_t, Retail_Rate_Name_Input);

% Import Solar PV Generation Profile Data
% Scale base 10-kW or 100-kW profile to match user-input PV system size
if Model_Type_Input == "Solar Plus Storage"
    [Solar_Profile_Master_Index, Solar_Profile_Description, Solar_PV_Profile_Data] = ...
        Import_Solar_PV_Profile_Data(Input_Output_Data_Directory_Location, OSESMO_Git_Repo_Directory, delta_t, ...
    Solar_Profile_Name_Input, Solar_Size_Input);    
    
elseif Model_Type_Input == "Storage Only" || Solar_Profile_Name_Input == "No Solar" 
    Solar_Profile_Master_Index = "";
    Solar_Profile_Description = "";
    Solar_PV_Profile_Data = zeros(size(Load_Profile_Data));   
end


% Import Utility Marginal Cost Data
% Marginal Costs are mapped to load profile location
[Generation_Cost_Data, Representative_Distribution_Cost_Data] = ...
Import_Utility_Marginal_Cost_Data(Input_Output_Data_Directory_Location, ...
OSESMO_Git_Repo_Directory, delta_t, Load_Profile_Name_Input);


% Set Directory to Box Sync Folder
cd(Input_Output_Data_Directory_Location)


%% Iterate Through Months & Filter Data to Selected Month

% Initialize Blank Variables to store optimal decision variable values for
% all months

% Initialize Decision Variable Vectors
P_ES_in = [];

P_ES_out = [];

Ene_Lvl = [];

P_max_NC = [];

P_special_max = [];
    
P_max_peak = [];
    
P_max_part_peak = [];
    

% Initialize Monthly Cost Variable Vectors
Fixed_Charge_Vector = [];

SM_DC_Baseline_Vector = [];
SM_DC_with_Solar_Only_Vector = [];
SM_DC_with_Solar_and_Storage_Vector = [];

NC_DC_Baseline_Vector = [];
NC_DC_with_Solar_Only_Vector = [];
NC_DC_with_Solar_and_Storage_Vector = [];

CPK_DC_Baseline_Vector = [];
CPK_DC_with_Solar_Only_Vector = [];
CPK_DC_with_Solar_and_Storage_Vector = [];

CPP_DC_Baseline_Vector = [];
CPP_DC_with_Solar_Only_Vector = [];
CPP_DC_with_Solar_and_Storage_Vector = [];

Energy_Charge_Baseline_Vector = [];
Energy_Charge_with_Solar_Only_Vector = [];
Energy_Charge_with_Solar_and_Storage_Vector = [];

Cycles_Vector = [];
Cycling_Penalty_Vector = [];


for Month_Iter = 1:12 % Iterate through all months
    
    % Filter Load Profile Data to Selected Month
    Load_Profile_Data_Month = Load_Profile_Data(Month_Data == Month_Iter, :);
    
    % Filter PV Production Profile Data to Selected Month
    Solar_PV_Profile_Data_Month = Solar_PV_Profile_Data(Month_Data == Month_Iter, :);
    
    % Filter Volumetric Rate Data to Selected Month
    Volumetric_Rate_Data_Month = Volumetric_Rate_Data(Month_Data == Month_Iter, :);
    
    % Filter Marginal Emissions Data to Selected Month
    Marginal_Emissions_Rate_Forecast_Data_Month = ...
        Marginal_Emissions_Rate_Forecast_Data(Month_Data == Month_Iter, :);
    
    % Filter Carbon Adder Data to Selected Month
    Carbon_Adder_Data_Month = Carbon_Adder_Data(Month_Data == Month_Iter, :);
    
    % Set Demand Charge Values Based on Month
    
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)      
        Peak_DC = Summer_Peak_DC;
        Part_Peak_DC = Summer_Part_Peak_DC;
        Special_Maximum_DC = Summer_Special_Maximum_DC;
        Noncoincident_DC = Summer_Noncoincident_DC;
        
        Peak_DC_Period = Summer_Peak_DC_Period;
        Part_Peak_DC_Period = Summer_Part_Peak_DC_Period;
        Special_Maximum_DC_Period = Summer_Special_Maximum_DC_Period;
        Noncoincident_DC_Period = Summer_Noncoincident_DC_Period;
        
    else        
        Peak_DC = Winter_Peak_DC;
        Part_Peak_DC = Winter_Part_Peak_DC;
        Special_Maximum_DC = Winter_Special_Maximum_DC;
        Noncoincident_DC = Winter_Noncoincident_DC;
        
        Peak_DC_Period = Winter_Peak_DC_Period;
        Part_Peak_DC_Period = Winter_Part_Peak_DC_Period;
        Special_Maximum_DC_Period = Winter_Special_Maximum_DC_Period;
        Noncoincident_DC_Period = Winter_Noncoincident_DC_Period;       
    end
    
    
    % Filter Peak and Part-Peak Binary Data to Selected Month
    
    if Summer_Peak_DC > 0
        Summer_Peak_Binary_Data_Month = Summer_Peak_Binary_Data(Month_Data == Month_Iter, :);
    end
    
    if Summer_Part_Peak_DC > 0
        Summer_Part_Peak_Binary_Data_Month = Summer_Part_Peak_Binary_Data(Month_Data == Month_Iter, :);
    end
    
    if Winter_Peak_DC > 0
        Winter_Peak_Binary_Data_Month = Winter_Peak_Binary_Data(Month_Data == Month_Iter, :);
    end
    
    if Winter_Part_Peak_DC > 0
        Winter_Part_Peak_Binary_Data_Month = Winter_Part_Peak_Binary_Data(Month_Data == Month_Iter, :);
    end
    
    if Special_Maximum_DC > 0
        Special_Maximum_Demand_Binary_Data_Month = Special_Maximum_Demand_Binary_Data(Month_Data == Month_Iter, :);
    end
    
    % Filter Day Data to Selected Month
    if length(Day_Data) > 0
        Day_Data_Month = Day_Data(Month_Data == Month_Iter, :);
    end
       
    %% Add "Padding" to Every Month of Data
    % Don't pad Month 12, because the final state of charge is constrained
    % to equal the original state of charge.
    
    if any(Month_Iter == 1:11)
    
        % Pad Load Profile Data
        Load_Profile_Data_Month_Padded = [Load_Profile_Data_Month;
            Load_Profile_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end)];
        
        % Pad PV Production Profile Data
        Solar_PV_Profile_Data_Month_Padded = [Solar_PV_Profile_Data_Month;
            Solar_PV_Profile_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end)];
        
        % Pad Volumetric Rate Data
        Volumetric_Rate_Data_Month_Padded = [Volumetric_Rate_Data_Month;
            Volumetric_Rate_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end)];
        
        % Pad Marginal Emissions Data
        Marginal_Emissions_Rate_Data_Month_Padded = [Marginal_Emissions_Rate_Forecast_Data_Month;
            Marginal_Emissions_Rate_Forecast_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end)];
        
        % Pad Carbon Adder Data 
        Carbon_Adder_Data_Month_Padded = [Carbon_Adder_Data_Month;
            Carbon_Adder_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end)];
        
        
        % Pad Peak and Part-Peak Binary Data
        
        if Summer_Peak_DC > 0
            Summer_Peak_Binary_Data_Month_Padded = [Summer_Peak_Binary_Data_Month;
                Summer_Peak_Binary_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end)];
        end
        
        if Summer_Part_Peak_DC > 0
            Summer_Part_Peak_Binary_Data_Month_Padded = [Summer_Part_Peak_Binary_Data_Month;
                Summer_Part_Peak_Binary_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end)];
        end
        
        if Winter_Peak_DC > 0
            Winter_Peak_Binary_Data_Month_Padded = [Winter_Peak_Binary_Data_Month;
                Winter_Peak_Binary_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end)];
        end
        
        if Winter_Part_Peak_DC > 0
            Winter_Part_Peak_Binary_Data_Month_Padded = [Winter_Part_Peak_Binary_Data_Month;
                Winter_Part_Peak_Binary_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end)];
        end
        
        if Special_Maximum_DC > 0
            Special_Maximum_Demand_Binary_Data_Month_Padded = [Special_Maximum_Demand_Binary_Data_Month;
                Special_Maximum_Demand_Binary_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end)];
        end
        
        
        % Pad Day Data (ex. add days "32", "33", and "34")
        if length(Day_Data) > 0
            Day_Data_Month_Padding = Day_Data_Month(end-(End_of_Month_Padding_Days * 24 * (1/delta_t) - 1):end);
            Day_Data_Month_Padding = Day_Data_Month_Padding + 3;
            Day_Data_Month_Padded = [Day_Data_Month; Day_Data_Month_Padding];
        end
    
        
    elseif Month_Iter == 12
        
        % Don't Pad Load Profile Data
        Load_Profile_Data_Month_Padded = Load_Profile_Data_Month;
        
        % Don't Pad PV Production Profile Data
        Solar_PV_Profile_Data_Month_Padded = Solar_PV_Profile_Data_Month;
        
        % Don't Pad Volumetric Rate Data
        Volumetric_Rate_Data_Month_Padded = Volumetric_Rate_Data_Month;
        
        % Don't Pad Marginal Emissions Data
        Marginal_Emissions_Rate_Data_Month_Padded = Marginal_Emissions_Rate_Forecast_Data_Month;
        
        % Don't Pad Carbon Adder Data
        Carbon_Adder_Data_Month_Padded = Carbon_Adder_Data_Month;
        
        % Don't Pad Peak and Part-Peak Binary Data
        
        if Summer_Peak_DC > 0
            Summer_Peak_Binary_Data_Month_Padded = Summer_Peak_Binary_Data_Month;
        end
        
        if Summer_Part_Peak_DC > 0
            Summer_Part_Peak_Binary_Data_Month_Padded = Summer_Part_Peak_Binary_Data_Month;
        end
        
        if Winter_Peak_DC > 0
            Winter_Peak_Binary_Data_Month_Padded = Winter_Peak_Binary_Data_Month;
        end
        
        if Winter_Part_Peak_DC > 0
            Winter_Part_Peak_Binary_Data_Month_Padded = Winter_Part_Peak_Binary_Data_Month;
        end
        
        if Special_Maximum_DC > 0
            Special_Maximum_Demand_Binary_Data_Month_Padded = Special_Maximum_Demand_Binary_Data_Month;
        end
        
    end
    
  
    
    %% Initialize Cost Vector "c"
    
    % nts = numtsteps = number of timesteps
    numtsteps = length(Load_Profile_Data_Month_Padded);
    all_tsteps = linspace(1,numtsteps, numtsteps)';
    numtsteps_unpadded = length(Load_Profile_Data_Month);
    
    if Noncoincident_DC_Period == "Monthly"
        P_max_NC_Indices = 3 * numtsteps + 1;
    elseif Noncoincdient_DC_Period == "Daily"
        P_max_NC_Indices = 3 * numtsteps + (1:Day_Data_Month_Padded(end));     
    end
    
    if Special_Maximum_DC_Period == "Monthly"
        P_special_max_Indices = P_max_NC_Indices(end) + 1;
    elseif Special_Maximum_DC_Period == "Daily"
        P_special_max_Indices = P_max_NC_Indices(end) + (1:Day_Data_Month_Padded(end));
    end
    
    if Peak_DC_Period == "Monthly"
        P_max_peak_Indices = P_special_max_Indices(end) + 1;
    elseif Peak_DC_Period == "Daily"
        P_max_peak_Indices = P_special_max_Indices(end) + (1:Day_Data_Month_Padded(end));
    end
    
    if Part_Peak_DC_Period == "Monthly"
        P_max_part_peak_Indices = P_max_peak_Indices(end) + 1;
    elseif Part_Peak_DC_Period == "Daily"
        P_max_part_peak_Indices = P_max_peak_Indices(end) + (1:Day_Data_Month_Padded(end));
    end
    
    
    % x = [P_ES_in_grid(size nts); P_ES_out(size nts); Ene_Lvl(size nts);...
    % P_max_NC (size 1 or ~30); P_special_max(size 1 or ~30); P_max_peak (size 1 or ~30); P_max_part_peak (size 1 or ~30); ];
    
    % Even if the system is charging from solar, it still has a relative cost
    % equal to the cost of grid power (Volumetric Rate).
    % This is because every amount of PV power going into the battery is
    % not used to offset load or export to the grid.
    
    c_Month_Bill_Only = [(Volumetric_Rate_Data_Month_Padded * delta_t); ...
        (-Volumetric_Rate_Data_Month_Padded * delta_t); ...
        zeros(numtsteps, 1);
        Noncoincident_DC * ones(length(P_max_NC_Indices), 1);
        Special_Maximum_DC * ones(length(P_special_max_Indices), 1);
        Peak_DC * ones(length(P_max_peak_Indices), 1);
        Part_Peak_DC * ones(length(P_max_part_peak_Indices), 1)];    
    
    % The same is true of carbon emissions. Every amount of PV power going into the battery is
    % not used at that time to offset emissions from the load or from the grid.
        
    c_Month_Carbon_Only = [(Carbon_Adder_Data_Month_Padded * delta_t); ...
        (-Carbon_Adder_Data_Month_Padded * delta_t); ...
        zeros(numtsteps, 1);
        zeros(length(P_max_NC_Indices), 1);
        zeros(length(P_special_max_Indices), 1);
        zeros(length(P_max_peak_Indices), 1);
        zeros(length(P_max_part_peak_Indices), 1)];
    
    c_Month_Degradation_Only = [(((Eff_c * cycle_pen)/(2 * Total_Storage_Capacity)) * delta_t) * ones(numtsteps,1); ...
        ((cycle_pen/(Eff_d * 2 * Total_Storage_Capacity)) * delta_t) * ones(numtsteps,1); ...
        zeros(numtsteps, 1);
        zeros(length(P_max_NC_Indices), 1);
        zeros(length(P_special_max_Indices), 1);
        zeros(length(P_max_peak_Indices), 1);
        zeros(length(P_max_part_peak_Indices), 1)];
    
    c_Month = c_Month_Bill_Only + c_Month_Carbon_Only + c_Month_Degradation_Only;
    
    % This is the length of the vectors c and x, or the total number of decision variables.
    length_x = length(c_Month);
    
    
    %% Decision Variable Indices
    
    % P_ES_in = x(1:numtsteps);
    % P_ES_out = x(numtsteps+1:2*numtsteps);
    % Ene_Lvl = x(2*numtsteps+1:3*numtsteps);
    % P_max_NC = x(P_max_NC_Indices);
    % P_special_max = x(P_special_max_Indices);
    % P_max_peak = x(P_max_peak_Indices);
    % P_max_part_peak = x(P_max_part_peak_Indices);
    
    %% State of Charge Constraint
    
    % This constraint represents conservation of energy as it flows into and out of the
    % energy storage system, while accounting for efficiency losses.
    
    %For t in [0, numsteps-1]:
    
    % E(t+1) = E(t) + [Eff_c*P_ES_in(t) - (1/Eff_d)*P_ES_out(t)] * delta_t
    
    % E(t) - E(t+1) + Eff_c*P_ES_in(t) * delta_t - (1/Eff_d)*P_ES_out(t) * delta_t = 0
    
    % An equality constraint can be transformed into two inequality constraints
    % Ax = 0 -> Ax <=0 , -Ax <=0
    
    % Number of rows in each inequality constraint matrix = (numtsteps - 1)
    % Number of columns in each inequality constraint matrix = number of
    % decision variables = length_x
    
    A_E = sparse(numtsteps-1,length_x);
    b_E = sparse(numtsteps-1,1);
    
    for n = 1:(numtsteps-1)
        A_E(n, n + (2 * numtsteps)) = 1;
        A_E(n, n + (2 * numtsteps) + 1) = -1;
        A_E(n, n) = Eff_c * delta_t;
        A_E(n, n + numtsteps) = (-1/Eff_d) * delta_t;
    end
    
    A_Month = [A_E;-A_E];
    
    b_Month = [b_E;-b_E];
    
    
    
    %% Energy Storage Charging Power Constraint
    
    % This constraint sets maximum and minimum values for P_ES_in.
    % The minimum is 0 kW, and the maximum is Storage_Power_Rating_Input.
    
    % P_ES_in >= 0 -> -P_ES_in <= 0
    
    % P_ES_in <= Storage_Power_Rating_Input
    
    % Number of rows in inequality constraint matrix = numtsteps
    % Number of columns in inequality constraint matrix = length_x
    A_P_ES_in = sparse(numtsteps, length_x);
    
    for n = 1:numtsteps
        A_P_ES_in(n, n) = -1;
    end
    
    A_Month = [A_Month; A_P_ES_in; -A_P_ES_in];
    
    b_Month = [b_Month; sparse(numtsteps,1); Storage_Power_Rating_Input * ones(numtsteps,1)];
    
    %% Energy Storage Discharging Power Constraint
    
    % This constraint sets maximum and minimum values for P_ES_out.
    % The minimum is 0 kW, and the maximum is Storage_Power_Rating_Input.
    
    % P_ES_out >= 0 -> -P_ES_out <= 0
    
    % P_ES_out <= Storage_Power_Rating_Input
    
    A_P_ES_out = sparse(numtsteps, length_x);
    
    for n = 1:numtsteps
        A_P_ES_out(n, n + numtsteps) = -1;
    end
    
    A_Month = [A_Month; A_P_ES_out; -A_P_ES_out];
    
    b_Month = [b_Month; sparse(numtsteps,1); Storage_Power_Rating_Input * ones(numtsteps,1)];
    
    %% State of Charge Minimum/Minimum Constraints
    
    % This constraint sets maximum and minimum values on the Energy Level.
    % The minimum value is 0, and the maximum value is Usable_Storage_Capacity, the size of the
    % battery. Note: this optimization defines the range [0, Usable_Storage_Capacity] as the
    % effective storage capacity of the battery, without accounting for
    % depth of discharge.
    
    % Ene_Lvl(t) >= 0 -> -Ene_Lvl(t) <=0
    
    A_Ene_Lvl_min = sparse(numtsteps, length_x);
    b_Ene_Lvl_min = sparse(numtsteps, 1);
    
    for n = 1:numtsteps
        A_Ene_Lvl_min(n, n + (2 * numtsteps)) = -1;
    end
    
    A_Month = [A_Month;A_Ene_Lvl_min];
    b_Month = [b_Month;b_Ene_Lvl_min];
    
    
    % Ene_Lvl(t) <= Size_ES
    
    A_Ene_Lvl_max = sparse(numtsteps, length_x);
    b_Ene_Lvl_max = Usable_Storage_Capacity * ones(numtsteps,1);
    
    for n = 1:numtsteps
        A_Ene_Lvl_max(n, n + (2 * numtsteps)) = 1;
    end
    
    A_Month = [A_Month; A_Ene_Lvl_max];
    
    b_Month = [b_Month; b_Ene_Lvl_max];
    
    %% Initial State of Charge Constraint
    
    % In the first month, this constraint initializes the energy level of the battery at
    % a user-defined percentage of the original battery capacity.
    % In all other month, this constraints initializes the energy level of
    % the battery at the final battery level from the previous month.
    
    % E(0) = Initial_Final_SOC * Usable_Storage_Capacity_Input
    % E(0) <= Initial_Final_SOC * Usable_Storage_Capacity_Input, -E(0) <= Initial_Final_SOC * Usable_Storage_Capacity_Input
    
    % E(0) = Previous_Month_Final_Energy_Level
    % E(0) <= Previous_Month_Final_Energy_Level, -E(0) <= Previous_Month_Final_Energy_Level  
    
    
    A_Ene_Lvl_0 = sparse(1, length_x);
    
    A_Ene_Lvl_0(1, (2*numtsteps) + 1) = 1;
    
    if Month_Iter == 1
        
        b_Ene_Lvl_0 = Initial_Final_SOC * Usable_Storage_Capacity_Input;
        
    elseif any(Month_Iter == 2:12)
        
        b_Ene_Lvl_0 = Next_Month_Initial_Energy_Level;
        
    end
    
    A_Month = [A_Month; A_Ene_Lvl_0; -A_Ene_Lvl_0];
    
    b_Month = [b_Month; b_Ene_Lvl_0; -b_Ene_Lvl_0];
    
    %% Final State of Charge Constraints
    
    % This constraint fixes the final state of charge of the battery at a user-defined percentage
    % of the original battery capacity,
    % to prevent it from discharging completely in the final timesteps.
    
    % E(N) = Initial_Final_SOC * Usable_Storage_Capacity_Input
    % E(N) <= Initial_Final_SOC * Usable_Storage_Capacity_Input, -E(N) <= Initial_Final_SOC * Usable_Storage_Capacity_Input
    
    A_Ene_Lvl_N = sparse(1, length_x);
    
    A_Ene_Lvl_N(1, 3 * numtsteps) = 1;
    
    b_Ene_Lvl_N = Initial_Final_SOC * Usable_Storage_Capacity_Input;
    
    A_Month = [A_Month; A_Ene_Lvl_N; -A_Ene_Lvl_N];
    
    b_Month = [b_Month; b_Ene_Lvl_N; -b_Ene_Lvl_N];
    
    
    %% Noncoincident Demand Charge Constraint
    % Note: this logic only works for monthly demand charges.
    
    % This constraint linearizes the noncoincident demand charge constraint.
    % Setting the demand charge value as a decision variable incentivizes
    % "demand capping" to reduce the value of max(P_load(t)) to an optimal
    % level without using the nonlinear max() operator.
    % The noncoincident demand charge applies across all 15-minute intervals.
    
    % P_load(t) - P_PV(t) + P_ES_in(t) - P_ES_out(t) <= P_max_NC for all t
    % P_ES_in(t) - P_ES_out(t) - P_max_NC <= - P_load(t) + P_PV(t) for all t
    
    if Noncoincident_DC > 0
        
        A_NC_DC = sparse(numtsteps, length_x);
        b_NC_DC = -Load_Profile_Data_Month_Padded + Solar_PV_Profile_Data_Month_Padded;
        
        for n = 1:numtsteps
            A_NC_DC(n, n) = 1;
            A_NC_DC(n, n + numtsteps) = -1;
            A_NC_DC(n, P_max_NC_Indices) = -1;
            
        end
        
        A_Month = [A_Month; A_NC_DC];
        b_Month = [b_Month; b_NC_DC];
        
    end
    
    % Add P_max_NC >=0 Constraint
    % -P_max_NC <= 0
    % Note: this non-negativity constraint is added even if the noncoincident
    % demand charge is $0/kW for this tariff. This ensures that the
    % decision variable P_max_NC goes to zero, and is not negative.
    
    A_NC_DC_gt0 = sparse(1, length_x);
    A_NC_DC_gt0(1, P_max_NC_Indices) = -1;
    b_NC_DC_gt0 = 0;
    
    A_Month = [A_Month; A_NC_DC_gt0];
    b_Month = [b_Month; b_NC_DC_gt0];
    
    
    %% Special Maximum Demand Charge Constraint
    % Note: this logic only works for monthly demand charges.
    
    % This constraint linearizes the special maximum demand charge constraint.
    % This demand charge only applies for the hours where this demand charge is active (all hours except 9:00 am - 2:00 pm for PG&E B-19 Option S).
    
    % P_load(t) - P_PV(t) + P_ES_in(t) - P_ES_out(t) <= P_special_max for Special Maximum Demand Charge t only
    % P_ES_in(t) - P_ES_out(t) - P_special_max <= - P_load(t) + P_PV(t) for Special Maximum Demand Charge t only
    
    if Special_Maximum_DC > 0
        
        Special_Maximum_Indices = all_tsteps(Special_Maximum_Demand_Binary_Data_Month_Padded == 1, :);
        A_SM_DC = sparse(sum(Special_Maximum_Demand_Binary_Data_Month_Padded), length_x);
        b_SM_DC = -Load_Profile_Data_Month_Padded(Special_Maximum_Demand_Binary_Data_Month_Padded == 1, :) + ...
            Solar_PV_Profile_Data_Month_Padded(Special_Maximum_Demand_Binary_Data_Month_Padded == 1, :);
        
        for n = 1:length(Special_Maximum_Indices)
            A_SM_DC(n, Special_Maximum_Indices(n)) = 1; % P_ES_in(t in Special_Maximum_Indices)
            A_SM_DC(n, numtsteps + Special_Maximum_Indices(n)) = -1; % P_ES_out(t in Special_Maximum_Indices)
            A_SM_DC(n, P_special_max_Indices) = -1; % P_special_max
        end
        
        A_Month = [A_Month; A_SM_DC];
        b_Month = [b_Month; b_SM_DC];
        
    end
    
    
    % Add P_special_max >=0 Constraint
    % -P_special_max <= 0
    % Note: this non-negativity constraint is added even if the coincident peak
    % demand charge is $0/kW for this tariff. This ensures that the
    % decision variable P_max_peak goes to zero, and is not negative.
    
    A_SM_DC_gt0 = sparse(1, length_x);
    A_SM_DC_gt0(1, P_special_max_Indices) = -1;
    b_SM_DC_gt0 = 0;
    
    A_Month = [A_Month; A_SM_DC_gt0];
    b_Month = [b_Month; b_SM_DC_gt0];
    
    %% Coincident Peak Demand Charge Constraint
    
    % This constraint linearizes the coincident peak demand charge constraint.
    % This demand charge only applies for peak hours.
    
    % P_load(t) - P_PV(t) + P_ES_in(t) - P_ES_out(t) <= P_max_peak for Peak t only
    % P_ES_in(t) - P_ES_out(t) - P_max_peak <= - P_load(t) + P_PV(t) for Peak t only
    
    if Peak_DC > 0
        
        if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
            Peak_Indices = all_tsteps(Summer_Peak_Binary_Data_Month_Padded == 1, :);
            A_CPK_DC = sparse(sum(Summer_Peak_Binary_Data_Month_Padded), length_x);
            b_CPK_DC = -Load_Profile_Data_Month_Padded(Summer_Peak_Binary_Data_Month_Padded == 1, :) + ...
                Solar_PV_Profile_Data_Month_Padded(Summer_Peak_Binary_Data_Month_Padded == 1, :);
        else
            Peak_Indices = all_tsteps(Winter_Peak_Binary_Data_Month_Padded == 1, :);
            A_CPK_DC = sparse(sum(Winter_Peak_Binary_Data_Month_Padded), length_x);
            b_CPK_DC = -Load_Profile_Data_Month_Padded(Winter_Peak_Binary_Data_Month_Padded == 1, :) + ...
                Solar_PV_Profile_Data_Month_Padded(Winter_Peak_Binary_Data_Month_Padded == 1, :);
        end
        
        if Peak_DC_Period == "Monthly"
            for n = 1:length(Peak_Indices)
                A_CPK_DC(n, Peak_Indices(n)) = 1; % P_ES_in(t in Peak_Indices)
                A_CPK_DC(n, numtsteps + Peak_Indices(n)) = -1; % P_ES_out(t in Peak_Indices)
                A_CPK_DC(n, P_max_peak_Indices) = -1; % P_max_peak (single decision variable for whole month)
            end
            
        elseif Peak_DC_Period == "Daily"
            for n = 1:length(Peak_Indices)
                A_CPK_DC(n, Peak_Indices(n)) = 1; % P_ES_in(t in Peak_Indices)
                A_CPK_DC(n, numtsteps + Peak_Indices(n)) = -1; % P_ES_out(t in Peak_Indices)
                A_CPK_DC(n, P_max_peak_Indices(Day_Data_Month_Padded(Peak_Indices(n)))) = -1; % P_max_peak (different decision variable for each day)
            end
            
        end
        
        A_Month = [A_Month; A_CPK_DC];
        b_Month = [b_Month; b_CPK_DC];
        
    end
    
    
    % Add P_max_peak >=0 Constraint
    % -P_max_peak <= 0
    % Note: this non-negativity constraint is added even if the coincident peak
    % demand charge is $0/kW for this tariff. This ensures that the
    % decision variable P_max_peak goes to zero, and is not negative.
    
    if Peak_DC_Period == "Monthly"
        A_CPK_DC_gt0 = sparse(1, length_x);
        A_CPK_DC_gt0(1, P_max_peak_Indices) = -1;
        b_CPK_DC_gt0 = 0;
        
    elseif Peak_DC_Period == "Daily"
        A_CPK_DC_gt0 = sparse(length(P_max_peak_Indices), length_x); 
        for n = 1:length(P_max_peak_Indices)
            A_CPK_DC_gt0(n, P_max_peak_Indices(n)) = -1;
        end
        b_CPK_DC_gt0 = zeros(length(P_max_peak_Indices), 1);
        
    end
    
    A_Month = [A_Month; A_CPK_DC_gt0];
    b_Month = [b_Month; b_CPK_DC_gt0];
    
    
    %% Coincident Part-Peak Demand Charge Constraint
    
    % This constraint linearizes the coincident part-peak demand charge
    % constraint.
    % This demand charge only applies for part-peak hours.
    
    % P_load(t) - P_PV(t) + P_ES_in(t) - P_ES_out(t) <= P_max_part_peak for Part-Peak t only
    % P_ES_in(t) - P_ES_out(t) - P_max_part_peak <= - P_load(t) + P_PV(t) for Part-Peak t only
    
    if Part_Peak_DC > 0
        
        if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
            Part_Peak_Indices = all_tsteps(Summer_Part_Peak_Binary_Data_Month_Padded == 1, :);
            A_CPP_DC = sparse(sum(Summer_Part_Peak_Binary_Data_Month_Padded), length_x);
            b_CPP_DC = -Load_Profile_Data_Month_Padded(Summer_Part_Peak_Binary_Data_Month_Padded == 1, :) + ...
                Solar_PV_Profile_Data_Month_Padded(Summer_Part_Peak_Binary_Data_Month_Padded == 1, :);      
        else
            Part_Peak_Indices = all_tsteps(Winter_Part_Peak_Binary_Data_Month_Padded == 1, :);
            A_CPP_DC = sparse(sum(Winter_Part_Peak_Binary_Data_Month_Padded), length_x);
            b_CPP_DC = -Load_Profile_Data_Month_Padded(Winter_Part_Peak_Binary_Data_Month_Padded == 1, :) + ...
                Solar_PV_Profile_Data_Month_Padded(Winter_Part_Peak_Binary_Data_Month_Padded == 1, :);
        end
        
        
        if Peak_DC_Period == "Monthly" 
            for n = 1:length(Part_Peak_Indices)
                A_CPP_DC(n, Part_Peak_Indices(n)) = 1; % P_ES_in(t in Part_Peak_Indices)
                A_CPP_DC(n, numtsteps + Part_Peak_Indices(n)) = -1; % P_ES_out(t in Part_Peak_Indices)
                A_CPP_DC(n, P_max_part_peak_Indices) = -1; % P_max_part_peak (single decision variable for whole month)
            end
            
        elseif Peak_DC_Period == "Daily"
            for n = 1:length(Part_Peak_Indices)
                A_CPP_DC(n, Part_Peak_Indices(n)) = 1; % P_ES_in(t in Part_Peak_Indices)
                A_CPP_DC(n, numtsteps + Part_Peak_Indices(n)) = -1; % P_ES_out(t in Part_Peak_Indices)
                A_CPP_DC(n, P_max_part_peak_Indices(Day_Data_Month_Padded(Part_Peak_Indices(n)))) = -1; % P_max_part_peak (different decision variable for each day)
            end
            
        end
        
        A_Month = [A_Month; A_CPP_DC];
        b_Month = [b_Month; b_CPP_DC];
        
        
    end
    
    % Add P_max_part_peak >=0 Constraint
    % -P_max_part_peak <= 0
    % Note: this non-negativity constraint is added even if the coincident part-peak
    % demand charge is $0/kW for this tariff. This ensures that the
    % decision variable P_max_part_peak goes to zero, and is not negative.
    
    if Peak_DC_Period == "Monthly"
        A_CPP_DC_gt0 = sparse(1, length_x);
        A_CPP_DC_gt0(1, P_max_part_peak_Indices) = -1;
        b_CPP_DC_gt0 = 0;
        
    elseif Peak_DC_Period == "Daily"
        A_CPP_DC_gt0 = sparse(length(P_max_part_peak_Indices), length_x);
        for n = 1:length(P_max_part_peak_Indices)
            A_CPP_DC_gt0(n, P_max_part_peak_Indices(n)) = -1;
        end
        b_CPP_DC_gt0 = zeros(length(P_max_part_peak_Indices), 1);
        
    end
    
    A_Month = [A_Month; A_CPP_DC_gt0];
    b_Month = [b_Month; b_CPP_DC_gt0];
    
    
%% Optional Constraint - Solar ITC Charging Constraint
    
    % This constraint requires that the storage system be charged 100% from
    % solar. This ensures that the customer receives 100% of the 
    % solar Incentive Tax Credit. The ITC amount is prorated by the amount 
    % of energy entering into the battery that comes from solar 
    % (ex. a storage system charged 90% from solar receives 90% of the ITC). 
    % As a result, the optimal amount of solar charging is likely higher
    % than the minimum requirement of 75%, and likely very close to 100%.

    % P_ES_in(t) <= P_PV(t)
    
    % Note that P_PV(t) can sometimes be negative for some PV profiles, if
    % the solar inverter is consuming energy at night. As a result, P_PV(t)
    % here refers to a modified version of the solar profile where all
    % negative values are set to 0. Otherwise, the model would break
    % because P_ES_in must be >= 0, and can't also be <= P_PV(t) if P_PV(t)
    % <= 0.
       
    
    if Model_Type_Input == "Solar Plus Storage" && Solar_Profile_Name_Input ~= "No Solar" && ...
            Solar_Size_Input > 0 && ITC_Constraint_Input == 1
        
        Solar_PV_Profile_Data_Month_Padded_Nonnegative = Solar_PV_Profile_Data_Month_Padded;
        Solar_PV_Profile_Data_Month_Padded_Nonnegative(Solar_PV_Profile_Data_Month_Padded_Nonnegative<0) = 0;
        
        A_ITC = sparse(numtsteps, length_x);
        b_ITC = Solar_PV_Profile_Data_Month_Padded_Nonnegative;
        
        for n = 1:numtsteps
            A_ITC(n, n) = 1;
        end
        
        A_Month = [A_Month; A_ITC];
        b_Month = [b_Month; b_ITC];
        
    end
    
    %% Optional Constraint - Equivalent Cycling Constraint
    
    % Note: due to the OSESMO model structure, the annual cycling requirement 
    % must be converted to an equivalent monthly cycling requirement.
    % All cycling must occur outside of the "padding" days at the end, which are removed after optimization.
    
    if Equivalent_Cycling_Constraint_Input > 0
           
        SGIP_Monthly_Cycling_Requirement = Equivalent_Cycling_Constraint_Input * (numtsteps_unpadded/length(Load_Profile_Data));
        
        % Formula for equivalent cycles is identical to the one used to calculate Cycles_Month:
        % Equivalent Cycles = sum((P_ES_in(t) * (((Eff_c)/(2 * Size_ES)) * delta_t)) + ...
        %    (P_ES_out(t) * ((1/(Eff_d * 2 * Size_ES)) * delta_t)));
        
        % Equivalent Cycles >= SGIP_Monthly_Cycling Requirement
        % To convert to standard linear program form, multiply both sides by -1.
        % -Equivalent Cycles <= -SGIP_Monthly_Cycling_Requirement
        
        A_Equivalent_Cycles = sparse(1, length_x);
        
        % sum of all P_ES_in(t) * (((Eff_c)/(2 * Size_ES)) * delta_t)
        A_Equivalent_Cycles(1, 1:numtsteps_unpadded) = -(((Eff_c)/(2 * Total_Storage_Capacity)) * delta_t);
        
        % sum of all P_ES_out(t) * ((1/(Eff_d * 2 * Size_ES)) * delta_t)
        A_Equivalent_Cycles(1, numtsteps+1:numtsteps+numtsteps_unpadded) = -((1/(Eff_d * 2 * Total_Storage_Capacity)) * delta_t);
        
        b_Equivalent_Cycles = -SGIP_Monthly_Cycling_Requirement;
        
        A_Month = [A_Month; A_Equivalent_Cycles];
        b_Month = [b_Month; b_Equivalent_Cycles];
    
    end
    
       
    %% Optional Constraint - No-Export Constraint
    
    % This constraint prevents the standalone energy-storage systems from
    % backfeeding power from the storage system onto the distribution grid.
    % Solar-plus storage systems are allowed to export to the grid.
    
    if Model_Type_Input == "Storage Only"
        
        % P_load(t) + P_ES_in(t) - P_ES_out(t) >= 0
        % -P_ES_in(t) + P_ES_out(t) <= P_load(t)
        
        A_No_Export = sparse(numtsteps, length_x);
        b_No_Export = Load_Profile_Data_Month_Padded;
        
        for n = 1:numtsteps
            A_No_Export(n, n) = -1;
            A_No_Export(n, n + numtsteps) = 1;
        end
        
        A_Month = [A_Month; A_No_Export];
        b_Month = [b_Month; b_No_Export];
        
    end
       
    
    %% Run LP Optimization Algorithm
    
    options = optimset('Display','none'); % Suppress "Optimal solution found" message.
    
    x_Month = linprog(c_Month,A_Month,b_Month, [], [], [], [], options); % Set Aeq, beq, LB, UB to []
    
    sprintf('Optimization complete for Month %d.', Month_Iter)
    
    %% Separate Decision Variable Vectors
    
    P_ES_in_Month_Padded = x_Month(1:numtsteps);
    
    P_ES_out_Month_Padded = x_Month(numtsteps+1:2*numtsteps);
    
    Ene_Lvl_Month_Padded = x_Month(2*numtsteps+1:3*numtsteps);
    
    P_max_NC_Month_Padded = x_Month(P_max_NC_Indices);
    P_special_max_Month_Padded = x_Month(P_special_max_Indices);
    P_max_peak_Month_Padded = x_Month(P_max_peak_Indices);
    P_max_part_peak_Month_Padded = x_Month(P_max_part_peak_Indices);
    
    
    %% Add Auxiliary Load/Parasitic Losses to P_ES_in
    
    P_ES_in_Month_Padded = P_ES_in_Month_Padded + Parasitic_Storage_Load;
    
    
    %% Remove "Padding" from Decision Variables
    
    % Data is padded in Months 1-11, and not in Month 12
    
    if any(Month_Iter == 1:11)
    
    P_ES_in_Month_Unpadded = P_ES_in_Month_Padded(1:(end-(End_of_Month_Padding_Days * 24 * (1/delta_t))));
    
    P_ES_out_Month_Unpadded = P_ES_out_Month_Padded(1:(end-(End_of_Month_Padding_Days * 24 * (1/delta_t))));
    
    Ene_Lvl_Month_Unpadded = Ene_Lvl_Month_Padded(1:(end-(End_of_Month_Padding_Days * 24 * (1/delta_t))));
    
    if Noncoincident_DC_Period == "Monthly"
        P_max_NC_Month_Unpadded = P_max_NC_Month_Padded;
    elseif Noncoincident_DC_Period == "Daily"
        P_max_NC_Month_Unpadded = P_max_NC_Month_Padded(1:(end-End_of_Month_Padding_Days));
    end
    
    if Special_Maximum_DC_Period == "Monthly"
        P_special_max_Month_Unpadded = P_special_max_Month_Padded;
    elseif Special_Maximum_DC_Period == "Daily"
        P_special_max_Month_Unpadded = P_special_max_Month_Padded(1:(end-End_of_Month_Padding_Days));
    end
    
    if Peak_DC_Period == "Monthly"
        P_max_peak_Month_Unpadded = P_max_peak_Month_Padded;
    elseif Peak_DC_Period == "Daily"
        P_max_peak_Month_Unpadded = P_max_peak_Month_Padded(1:(end-End_of_Month_Padding_Days));
    end
    
    if Part_Peak_DC_Period == "Monthly"
        P_max_part_peak_Month_Unpadded = P_max_part_peak_Month_Padded;
    elseif Part_Peak_DC_Period == "Daily"
        P_max_part_peak_Month_Unpadded = P_max_part_peak_Month_Padded(1:(end-End_of_Month_Padding_Days));
    end
    
    
    elseif Month_Iter == 12
        
        P_ES_in_Month_Unpadded = P_ES_in_Month_Padded;
        
        P_ES_out_Month_Unpadded = P_ES_out_Month_Padded;
        
        Ene_Lvl_Month_Unpadded = Ene_Lvl_Month_Padded;
        
        P_max_NC_Month_Unpadded = P_max_NC_Month_Padded;
        P_special_max_Month_Unpadded = P_special_max_Month_Padded;
        P_max_peak_Month_Unpadded = P_max_peak_Month_Padded;
        P_max_part_peak_Month_Unpadded = P_max_part_peak_Month_Padded;
        
    end
    
    % Save Final Energy Level of Battery for use in next month
    
    Previous_Month_Final_Energy_Level = Ene_Lvl_Month_Unpadded(length(Ene_Lvl_Month_Unpadded));
    
    Next_Month_Initial_Energy_Level = Previous_Month_Final_Energy_Level + ...
        ((Eff_c * P_ES_in_Month_Unpadded(length(P_ES_in_Month_Unpadded))) - ...
        ((1/Eff_d) * P_ES_out_Month_Unpadded(length(P_ES_out_Month_Unpadded)))) * delta_t;
    
    
    %% Calculate Peak Demand
        
    % Noncoincident Maximum Demand With and Without Storage
    % Note: this logic only works for monthly demand charges.
    P_max_NC_Month_Baseline = max(Load_Profile_Data_Month);
    P_max_NC_Month_with_Solar_Only = max(Load_Profile_Data_Month - Solar_PV_Profile_Data_Month);
    P_max_NC_Month_with_Solar_and_Storage = P_max_NC_Month_Unpadded;
    
    
    % Special Maximum Demand With and Without Storage
    % Note: this logic only works for monthly demand charges.
    if Special_Maximum_DC > 0
        P_special_max_Month_Baseline = max(Load_Profile_Data_Month(Special_Maximum_Demand_Binary_Data_Month == 1, :));
        
        P_special_max_Month_with_Solar_Only = max(Load_Profile_Data_Month(Special_Maximum_Demand_Binary_Data_Month == 1, :) - ...
            Solar_PV_Profile_Data_Month(Special_Maximum_Demand_Binary_Data_Month == 1, :));
        
        P_special_max_Month_with_Solar_and_Storage = P_special_max_Month_Unpadded;     
    else  
        % If there is no Coincident Peak Demand Period (or if the
        % corresponding demand charge is $0/kW), set P_max_CPK to 0 kW.
        P_special_max_Month_Baseline = 0;
        P_special_max_Month_with_Solar_Only = 0;
        P_special_max_Month_with_Solar_and_Storage = 0;     
    end
    
    
    % Coincident Peak Demand With and Without Storage
    
    if Peak_DC > 0
        
        if any(Month_Iter == First_Summer_Month:Last_Summer_Month) % Summer Months
            
            if Peak_DC_Period == "Monthly"
                P_max_CPK_Month_Baseline = max(Load_Profile_Data_Month(Summer_Peak_Binary_Data_Month == 1, :));
                
                P_max_CPK_Month_with_Solar_Only = max(Load_Profile_Data_Month(Summer_Peak_Binary_Data_Month == 1, :) - ...
                    Solar_PV_Profile_Data_Month(Summer_Peak_Binary_Data_Month == 1, :));
                
            elseif Peak_DC_Period == "Daily"
                P_max_CPK_Month_Baseline = [];
                P_max_CPK_Month_with_Solar_Only = [];
                
                for n = 1:Day_Data_Month(end)
                    Load_Profile_Data_Day = Load_Profile_Data_Month(Day_Data_Month == n);
                    Solar_PV_Profile_Data_Day = Solar_PV_Profile_Data_Month(Day_Data_Month == n);
                    Summer_Peak_Binary_Data_Day = Summer_Peak_Binary_Data_Month(Day_Data_Month == n);
                    P_max_CPK_Month_Baseline = [P_max_CPK_Month_Baseline; max(Load_Profile_Data_Day(Summer_Peak_Binary_Data_Day == 1, :))];
                    P_max_CPK_Month_with_Solar_Only = [P_max_CPK_Month_with_Solar_Only; max(Load_Profile_Data_Day(Summer_Peak_Binary_Data_Day == 1, :) - ...
                        Solar_PV_Profile_Data_Day(Summer_Peak_Binary_Data_Day == 1, :))];
                end
                
            end
            
        else % Winter Months
                   
            if Peak_DC_Period == "Monthly"
                P_max_CPK_Month_Baseline = max(Load_Profile_Data_Month(Winter_Peak_Binary_Data_Month == 1, :));

                P_max_CPK_Month_with_Solar_Only = max(Load_Profile_Data_Month(Winter_Peak_Binary_Data_Month == 1, :) - ...
                    Solar_PV_Profile_Data_Month(Winter_Peak_Binary_Data_Month == 1, :));
                     
            elseif Peak_DC_Period == "Daily"
                
                P_max_CPK_Month_Baseline = [];
                P_max_CPK_Month_with_Solar_Only = [];
                
                for n = 1:Day_Data_Month(end)
                          Load_Profile_Data_Day = Load_Profile_Data_Month(Day_Data_Month == n);
                          Solar_PV_Profile_Data_Day = Solar_PV_Profile_Data_Month(Day_Data_Month == n);
                          Winter_Peak_Binary_Data_Day = Winter_Peak_Binary_Data_Month(Day_Data_Month == n);
                          P_max_CPK_Month_Baseline = [P_max_CPK_Month_Baseline; max(Load_Profile_Data_Day(Winter_Peak_Binary_Data_Day == 1, :))];
                          P_max_CPK_Month_with_Solar_Only = [P_max_CPK_Month_with_Solar_Only; max(Load_Profile_Data_Day(Winter_Peak_Binary_Data_Day == 1, :) - ...
                    Solar_PV_Profile_Data_Day(Winter_Peak_Binary_Data_Day == 1, :))];  
                end
                
            end
            
            
        end
        
        P_max_CPK_Month_with_Solar_and_Storage = P_max_peak_Month_Unpadded;
        
    else
        
        % If there is no Coincident Peak Demand Period (or if the
        % corresponding demand charge is $0/kW), set P_max_CPK to 0 kW.
        P_max_CPK_Month_Baseline = 0;
        P_max_CPK_Month_with_Solar_Only = 0;
        P_max_CPK_Month_with_Solar_and_Storage = 0;
        
    end
    
    
    % Coincident Part-Peak Demand With and Without Storage
    
    if Part_Peak_DC > 0
        
        if any(Month_Iter == First_Summer_Month:Last_Summer_Month) % Summer Months
            
            if Part_Peak_DC_Period == "Monthly"
                P_max_CPP_Month_Baseline = max(Load_Profile_Data_Month(Summer_Part_Peak_Binary_Data_Month == 1, :));
                
                P_max_CPP_Month_with_Solar_Only = max(Load_Profile_Data_Month(Summer_Part_Peak_Binary_Data_Month == 1, :) - ...
                    Solar_PV_Profile_Data_Month(Summer_Part_Peak_Binary_Data_Month == 1, :));
                
            elseif Part_Peak_DC_Period == "Daily"
                P_max_CPP_Month_Baseline = [];
                P_max_CPP_Month_with_Solar_Only = [];
                
                for n = 1:Day_Data_Month(end)
                    Load_Profile_Data_Day = Load_Profile_Data_Month(Day_Data_Month == n);
                    Solar_PV_Profile_Data_Day = Solar_PV_Profile_Data_Month(Day_Data_Month == n);
                    Summer_Part_Peak_Binary_Data_Day = Summer_Part_Peak_Binary_Data_Month(Day_Data_Month == n);
                    P_max_CPP_Month_Baseline = [P_max_CPP_Month_Baseline; max(Load_Profile_Data_Day(Summer_Part_Peak_Binary_Data_Day == 1, :))];
                    P_max_CPP_Month_with_Solar_Only = [P_max_CPP_Month_with_Solar_Only; max(Load_Profile_Data_Day(Summer_Part_Peak_Binary_Data_Day == 1, :) - ...
                        Solar_PV_Profile_Data_Day(Summer_Part_Peak_Binary_Data_Day == 1, :))];
                end
                
            end
            
        else % Winter Months
            
            if Part_Peak_DC_Period == "Monthly"
                P_max_CPP_Month_Baseline = max(Load_Profile_Data_Month(Winter_Part_Peak_Binary_Data_Month == 1, :));
                
                P_max_CPP_Month_with_Solar_Only = max(Load_Profile_Data_Month(Winter_Part_Peak_Binary_Data_Month == 1, :) - ...
                    Solar_PV_Profile_Data_Month(Winter_Part_Peak_Binary_Data_Month == 1, :));
                
            elseif Part_Peak_DC_Period == "Daily"
                P_max_CPP_Month_Baseline = [];
                P_max_CPP_Month_with_Solar_Only = [];
                
                for n = 1:Day_Data_Month(end)
                    Load_Profile_Data_Day = Load_Profile_Data_Month(Day_Data_Month == n);
                    Solar_PV_Profile_Data_Day = Solar_PV_Profile_Data_Month(Day_Data_Month == n);
                    Winter_Part_Peak_Binary_Data_Day = Winter_Part_Peak_Binary_Data_Month(Day_Data_Month == n);
                    P_max_CPP_Month_Baseline = [P_max_CPP_Month_Baseline; max(Load_Profile_Data_Day(Winter_Part_Peak_Binary_Data_Day == 1, :))];
                    P_max_CPP_Month_with_Solar_Only = [P_max_CPP_Month_with_Solar_Only; max(Load_Profile_Data_Day(Winter_Part_Peak_Binary_Data_Day == 1, :) - ...
                        Solar_PV_Profile_Data_Day(Winter_Part_Peak_Binary_Data_Day == 1, :))];
                end
                
            end
            
        end
        
        P_max_CPP_Month_with_Solar_and_Storage = P_max_part_peak_Month_Unpadded;
        
    else
        
        % If there is no Coincident Part-Peak Demand Period (or if the
        % corresponding demand charge is $0/kW), set P_max_CPP to 0 kW.
        P_max_CPP_Month_Baseline = 0;
        P_max_CPP_Month_with_Solar_Only = 0;
        P_max_CPP_Month_with_Solar_and_Storage = 0;
        
    end
            
    
    %% Calculate Monthly Bill Cost with and Without Storage
    
    % Monthly Cost from Daily Fixed Charge
    % This value is not affected by the presence of storage.
    Fixed_Charge_Month = Fixed_Per_Meter_Month_Charge + (Fixed_Per_Meter_Day_Charge * length(Load_Profile_Data_Month)/(24 * (1/delta_t)));
    
    % Monthly Cost from Noncoincident Demand Charge - Baseline
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        NC_Demand_Charge_Month_Baseline = sum(Summer_Noncoincident_DC * P_max_NC_Month_Baseline);
    else
        NC_Demand_Charge_Month_Baseline = sum(Winter_Noncoincident_DC * P_max_NC_Month_Baseline);
    end

    % Monthly Cost from Noncoincident Demand Charge - With Solar Only
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        NC_Demand_Charge_Month_with_Solar_Only = sum(Summer_Noncoincident_DC * P_max_NC_Month_with_Solar_Only);
    else
        NC_Demand_Charge_Month_with_Solar_Only = sum(Winter_Noncoincident_DC * P_max_NC_Month_with_Solar_Only);
    end
    
    % Monthly Cost from Noncoincident Demand Charge - With Solar and Storage
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        NC_Demand_Charge_Month_with_Solar_and_Storage = sum(Summer_Noncoincident_DC * P_max_NC_Month_with_Solar_and_Storage);
    else
        NC_Demand_Charge_Month_with_Solar_and_Storage = sum(Winter_Noncoincident_DC * P_max_NC_Month_with_Solar_and_Storage);
    end
    
    
    % Monthly Cost from Special Maximum Demand Charge - Baseline
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        SM_Demand_Charge_Month_Baseline = sum(Summer_Special_Maximum_DC * P_special_max_Month_Baseline);
    else
        SM_Demand_Charge_Month_Baseline = sum(Winter_Special_Maximum_DC * P_special_max_Month_Baseline);
    end
    
    
    % Monthly Cost from Special Maximum Demand Charge - With Solar Only
    
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        SM_Demand_Charge_Month_with_Solar_Only = sum(Summer_Special_Maximum_DC * P_special_max_Month_with_Solar_Only);
    else
        SM_Demand_Charge_Month_with_Solar_Only = sum(Winter_Special_Maximum_DC * P_special_max_Month_with_Solar_Only);
    end
    
    
    % Monthly Cost from Special Maximum Demand Charge - With Solar and Storage
    
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        SM_Demand_Charge_Month_with_Solar_and_Storage = sum(Summer_Special_Maximum_DC * P_special_max_Month_with_Solar_and_Storage);
    else
        SM_Demand_Charge_Month_with_Solar_and_Storage = sum(Winter_Special_Maximum_DC * P_special_max_Month_with_Solar_and_Storage);
    end
    
    
    
    % Monthly Cost from Coincident Peak Demand Charge - Baseline
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        CPK_Demand_Charge_Month_Baseline = sum(Summer_Peak_DC * P_max_CPK_Month_Baseline);
    else
        CPK_Demand_Charge_Month_Baseline = sum(Winter_Peak_DC * P_max_CPK_Month_Baseline);
    end
    
    
    % Monthly Cost from Coincident Peak Demand Charge - With Solar Only
    
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        CPK_Demand_Charge_Month_with_Solar_Only = sum(Summer_Peak_DC * P_max_CPK_Month_with_Solar_Only);
    else
        CPK_Demand_Charge_Month_with_Solar_Only = sum(Winter_Peak_DC * P_max_CPK_Month_with_Solar_Only);
    end
    
    
    % Monthly Cost from Coincident Peak Demand Charge - With Solar and Storage
    
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        CPK_Demand_Charge_Month_with_Solar_and_Storage = sum(Summer_Peak_DC * P_max_CPK_Month_with_Solar_and_Storage);
    else
        CPK_Demand_Charge_Month_with_Solar_and_Storage = sum(Winter_Peak_DC * P_max_CPK_Month_with_Solar_and_Storage);
    end
    
    
    % Monthly Cost from Coincident Part-Peak Demand Charge - Baseline
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        CPP_Demand_Charge_Month_Baseline = sum(Summer_Part_Peak_DC * P_max_CPP_Month_Baseline);
    else
        CPP_Demand_Charge_Month_Baseline = sum(Winter_Part_Peak_DC * P_max_CPP_Month_Baseline);
    end
    

    % Monthly Cost from Coincident Part-Peak Demand Charge - With Solar Only
    
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        CPP_Demand_Charge_Month_with_Solar_Only = sum(Summer_Part_Peak_DC * P_max_CPP_Month_with_Solar_Only);
    else
        CPP_Demand_Charge_Month_with_Solar_Only = sum(Winter_Part_Peak_DC * P_max_CPP_Month_with_Solar_Only);
    end
    
    
    % Monthly Cost from Coincident Part-Peak Demand Charge - With Solar and Storage
    
    if any(Month_Iter == First_Summer_Month:Last_Summer_Month)
        CPP_Demand_Charge_Month_with_Solar_and_Storage = sum(Summer_Part_Peak_DC * P_max_CPP_Month_with_Solar_and_Storage);
    else
        CPP_Demand_Charge_Month_with_Solar_and_Storage = sum(Winter_Part_Peak_DC * P_max_CPP_Month_with_Solar_and_Storage);
    end
    
    
    % Monthly Cost from Volumetric Energy Rates - Baseline
    Energy_Charge_Month_Baseline = (Load_Profile_Data_Month' * Volumetric_Rate_Data_Month) * delta_t;
    
    % Monthly Cost from Volumetric Energy Rates - With Solar Only
    Solar_Only_Net_Load_Profile_Month = Load_Profile_Data_Month - Solar_PV_Profile_Data_Month;
    Energy_Charge_Month_with_Solar_Only = (Solar_Only_Net_Load_Profile_Month' * Volumetric_Rate_Data_Month) * delta_t;
    
    % Monthly Cost from Volumetric Energy Rates - With Solar and Storage
    Solar_Storage_Net_Load_Profile_Month = Load_Profile_Data_Month - Solar_PV_Profile_Data_Month + ...
                                           P_ES_in_Month_Unpadded - P_ES_out_Month_Unpadded;
    Energy_Charge_Month_with_Solar_and_Storage = (Solar_Storage_Net_Load_Profile_Month' * Volumetric_Rate_Data_Month) * delta_t;
    
    
    % Monthly Cycling Penalty
    
    Cycles_Month = sum((P_ES_in_Month_Unpadded * (((Eff_c)/(2 * Total_Storage_Capacity)) * delta_t)) + ...
        (P_ES_out_Month_Unpadded * ((1/(Eff_d * 2 * Total_Storage_Capacity)) * delta_t)));
    
    Cycling_Penalty_Month = sum((P_ES_in_Month_Unpadded * (((Eff_c * cycle_pen)/(2 * Total_Storage_Capacity)) * delta_t)) + ...
        (P_ES_out_Month_Unpadded * ((cycle_pen/(Eff_d * 2 * Total_Storage_Capacity)) * delta_t)));
    
    
    %% Update Battery Capacity Based on Monthly Cycling
    % This is to account for capacity fade in lithium-ion batteries.
    % Based on standard definitions of battery cycle life, lithium-ion batteries are
    % defined to have experienced capacity fade to 80% of its original
    % capacity by the end of its cycle life.
    % Flow batteries do not experience capacity fade.
    
    if Storage_Type_Input == "Lithium-Ion Battery"
        
        Usable_Storage_Capacity = Usable_Storage_Capacity - (Usable_Storage_Capacity_Input * (Cycles_Month/Cycle_Life) * 0.2);
        
    elseif Storage_Type_Input == "Flow Battery"
        
        Usable_Storage_Capacity = Usable_Storage_Capacity;
        
    end
    
    % Update Previous Month Final Energy Level to account for capacity fade, if battery is full at end
    % of month. Otherwise, optimization is infeasible.
    
    if Next_Month_Initial_Energy_Level > Usable_Storage_Capacity
        
        Next_Month_Initial_Energy_Level = Usable_Storage_Capacity;
        
    end
    
    
    %% Concatenate Decision Variable & Monthly Cost Values from Month Iteration
    
    % Decision Variable Concatenation
    P_ES_in = [P_ES_in; P_ES_in_Month_Unpadded];
    
    P_ES_out = [P_ES_out; P_ES_out_Month_Unpadded];
    
    Ene_Lvl = [Ene_Lvl; Ene_Lvl_Month_Unpadded];
    
    P_max_NC = [P_max_NC; P_max_NC_Month_with_Solar_and_Storage];
    
    P_special_max = [P_special_max; P_special_max_Month_with_Solar_and_Storage];
    
    P_max_peak = [P_max_peak; P_max_CPK_Month_with_Solar_and_Storage];
    
    P_max_part_peak = [P_max_part_peak; P_max_CPP_Month_with_Solar_and_Storage];
    
    
    % Monthly Cost Variable Concatenation
    Fixed_Charge_Vector = [Fixed_Charge_Vector; Fixed_Charge_Month];
    
    NC_DC_Baseline_Vector = [NC_DC_Baseline_Vector; NC_Demand_Charge_Month_Baseline];
    NC_DC_with_Solar_Only_Vector = [NC_DC_with_Solar_Only_Vector; NC_Demand_Charge_Month_with_Solar_Only];
    NC_DC_with_Solar_and_Storage_Vector = [NC_DC_with_Solar_and_Storage_Vector; NC_Demand_Charge_Month_with_Solar_and_Storage];
    
    SM_DC_Baseline_Vector = [SM_DC_Baseline_Vector; SM_Demand_Charge_Month_Baseline];
    SM_DC_with_Solar_Only_Vector = [SM_DC_with_Solar_Only_Vector; SM_Demand_Charge_Month_with_Solar_Only];
    SM_DC_with_Solar_and_Storage_Vector = [SM_DC_with_Solar_and_Storage_Vector; SM_Demand_Charge_Month_with_Solar_and_Storage];
    
    CPK_DC_Baseline_Vector = [CPK_DC_Baseline_Vector; CPK_Demand_Charge_Month_Baseline];
    CPK_DC_with_Solar_Only_Vector = [CPK_DC_with_Solar_Only_Vector; CPK_Demand_Charge_Month_with_Solar_Only];
    CPK_DC_with_Solar_and_Storage_Vector = [CPK_DC_with_Solar_and_Storage_Vector; CPK_Demand_Charge_Month_with_Solar_and_Storage];
    
    CPP_DC_Baseline_Vector = [CPP_DC_Baseline_Vector; CPP_Demand_Charge_Month_Baseline];
    CPP_DC_with_Solar_Only_Vector = [CPP_DC_with_Solar_Only_Vector; CPP_Demand_Charge_Month_with_Solar_Only];
    CPP_DC_with_Solar_and_Storage_Vector = [CPP_DC_with_Solar_and_Storage_Vector; CPP_Demand_Charge_Month_with_Solar_and_Storage];
    
    Energy_Charge_Baseline_Vector = [Energy_Charge_Baseline_Vector; Energy_Charge_Month_Baseline];
    Energy_Charge_with_Solar_Only_Vector = [Energy_Charge_with_Solar_Only_Vector; Energy_Charge_Month_with_Solar_Only];
    Energy_Charge_with_Solar_and_Storage_Vector = [Energy_Charge_with_Solar_and_Storage_Vector; Energy_Charge_Month_with_Solar_and_Storage];
    
    Cycles_Vector = [Cycles_Vector; Cycles_Month];
    
    Cycling_Penalty_Vector = [Cycling_Penalty_Vector; Cycling_Penalty_Month];
    
    
end

% Report total script runtime.

telapsed = toc(tstart);

sprintf('Model Run %0.f complete. Elapsed time to run the optimization model is %0.0f seconds.', Model_Run_Number_Input, telapsed)


%% Calculation of Additional Reported Model Inputs/Outputs

% Output current system date and time in standard ISO 8601 YYYY-MM-DD HH:MM format.
format shortg
Model_Run_Date_Time_Raw = clock;
Model_Run_Date_Time_Components = Model_Run_Date_Time_Raw(1:5); % Remove seconds column

Model_Run_Date_Time = "";

for Model_Run_Date_Time_Component_Iter = 1:length(Model_Run_Date_Time_Components)
   
    Model_Run_Date_Time_Component_Num = Model_Run_Date_Time_Components(Model_Run_Date_Time_Component_Iter);
    
    if Model_Run_Date_Time_Component_Num >= 10
        Model_Run_Date_Time_Component_String = num2str(Model_Run_Date_Time_Component_Num);
    else
        Model_Run_Date_Time_Component_String = ['0', num2str(Model_Run_Date_Time_Component_Num)];
    end
    
    if Model_Run_Date_Time_Component_Iter == 1 || Model_Run_Date_Time_Component_Iter == 2
       Model_Run_Date_Time = [Model_Run_Date_Time, Model_Run_Date_Time_Component_String, '-'];
    elseif Model_Run_Date_Time_Component_Iter == 3
        Model_Run_Date_Time = [Model_Run_Date_Time, Model_Run_Date_Time_Component_String, ' '];
    elseif Model_Run_Date_Time_Component_Iter == 4
        Model_Run_Date_Time = [Model_Run_Date_Time, Model_Run_Date_Time_Component_String, ':'];
    elseif Model_Run_Date_Time_Component_Iter == 5
        Model_Run_Date_Time = [Model_Run_Date_Time, Model_Run_Date_Time_Component_String];
    end
    
end

Model_Run_Date_Time = join(Model_Run_Date_Time, "");


% Convert Retail Rate Name Input (which contains both utility name and rate
% name) into Retail Rate Utility and Retail Rate Name Output

if contains(Retail_Rate_Name_Input, "PG&E")
    Retail_Rate_Utility = "PG&E";
elseif contains(Retail_Rate_Name_Input, "SCE")
    Retail_Rate_Utility = "SCE";
elseif contains(Retail_Rate_Name_Input, "SDG&E")
    Retail_Rate_Utility = "SDG&E";
end

Retail_Rate_Utility_Plus_Space = join([Retail_Rate_Utility, " "], "");

Retail_Rate_Name_Output = erase(Retail_Rate_Name_Input, Retail_Rate_Utility_Plus_Space);

% If Solar Profile Name is "No Solar", Solar Profile Name Output is Blank
if Solar_Profile_Name_Input == "No Solar"
    Solar_Profile_Name_Output = "";
else
    Solar_Profile_Name_Output = Solar_Profile_Name_Input;
end

% Storage Control Algorithm Description (Optional)
if Storage_Control_Algorithm_Name == "OSESMO Economic Dispatch"
    Storage_Control_Algorithm_Description = "Open Source Energy Storage Model - Economic Dispatch";
elseif Storage_Control_Algorithm_Name == "OSESMO Non-Economic Solar Self-Supply"
    Storage_Control_Algorithm_Description = "Open Source Energy Storage Model - Non-Economic Solar Self-Supply";
end

% Storage Algorithm Parameters Filename (Optional)
Storage_Control_Algorithms_Parameters_Filename = ""; % No storage parameters file.

% Other Incentives or Penalities (Optional)
Other_Incentives_or_Penalities = ""; % No other incentives or penalties.

Output_Summary_Filename = "OSESMO Reporting Inputs and Outputs.csv";

Output_Description_Filename = ""; % No output description file.

Output_Visualizations_Filename = "Multiple files - in same folder as Output Summary file."; % No single output visualizations file.

EV_Use = ""; % Model does not calculate or report EV usage information.

EV_Charge = ""; % Model does not calculate or report EV charge information.

EV_Gas_Savings = ""; % Model does not calculate or report EV gas savings information.

EV_GHG_Savings = ""; % Model does not calculate or report EV GHG savings information.




%% Output Directory/Folder Names

if ITC_Constraint_Input == 0
    ITC_Constraint_Folder_Name = "No ITC Constraint";
elseif ITC_Constraint_Input == 1
    ITC_Constraint_Folder_Name = "ITC Constraint";   
end

% Ensures that folder is called "No Emissions Forecast Signal",
% and not "No Emissions Forecast Signal Emissions Forecast Signal"

if Emissions_Forecast_Signal_Input == "No Emissions Forecast Signal"
    Emissions_Forecast_Signal_Input = "No";
end

Output_Directory_Filepath = "Model Outputs/" + ...
    Model_Type_Input + "/" + Model_Timestep_Resolution + "-Minute" + "/" + ...
    Load_Profile_Name_Input + "/" + Retail_Rate_Name_Input + "/" + ...
    Storage_Power_Rating_Input + " kW " + Usable_Storage_Capacity_Input + " kWh Storage/";

% Correct Emissions Forecast Signal Name back so that it is exported with
% the correct name in the Outputs model.

if Emissions_Forecast_Signal_Input == "No"
    Emissions_Forecast_Signal_Input = "No Emissions Forecast Signal";
end


% Create folder if one does not exist already

if ~exist(Output_Directory_Filepath, 'dir')
    Output_Directory_Filepath_Single_Quotes = char(Output_Directory_Filepath);
    mkdir(Output_Directory_Filepath_Single_Quotes); % mkdir only works with single-quote filepath
end


%% Plot Energy Storage Dispatch Schedule

numtsteps_year = length(Load_Profile_Data);

t = Start_Time_Input + linspace(0, ((numtsteps_year-1) * delta_t)/(24), numtsteps_year)';

P_ES = P_ES_out - P_ES_in;

if Show_Plots == 1 || Export_Plots ==1
    
    figure('NumberTitle', 'off')
    plot(t, P_ES,'r')
    xlim([t(1), t(end)])
    ylim([-Storage_Power_Rating_Input * 1.1, Storage_Power_Rating_Input * 1.1]) % Make ylim 10% larger than storage power rating.
    xlabel('Date & Time','FontSize',15);
    ylabel('Energy Storage Output (kW)','FontSize',15);
    title('Energy Storage Dispatch Profile','FontSize',15)     
  
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Storage Dispatch Plot.png");
        
        saveas(gcf, Output_Directory_Filepath + "Storage Dispatch Plot");
        
    end
    
end


%% Plot Energy Storage Energy Level

if Show_Plots == 1 || Export_Plots ==1
    
    figure('NumberTitle', 'off')
    plot(t, Ene_Lvl,'r')
    xlim([t(1), t(end)])
    ylim([-Usable_Storage_Capacity_Input * 0.1, Usable_Storage_Capacity_Input * 1.1]) % Make ylim 10% larger than energy storage level.
    xlabel('Date & Time','FontSize',15);
    ylabel('Energy Storage Energy Level (kWh)','FontSize',15);
    title('Energy Storage Energy Level','FontSize',15) 
    
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Energy Level Plot.png");
        
        saveas(gcf, Output_Directory_Filepath + "Energy Level Plot");
        
    end
    
end


%% Plot Volumetric Electricity Price Schedule and Marginal Carbon Emission Rates

if Show_Plots == 1 || Export_Plots ==1
    
%     figure('NumberTitle', 'off')
%     yyaxis left
%     plot(t, Volumetric_Rate_Data)
%     xlim([t(1), t(end)])
%     ylim([-max(Volumetric_Rate_Data) * 0.1, max(Volumetric_Rate_Data) * 1.1]) % Make ylim 10% larger than volumetric rate range.
%     xlabel('Date & Time','FontSize',15);
%     ylabel('Energy Price ($/kWh)', 'FontSize', 15);
%     yyaxis right
%     plot(t, Marginal_Emissions_Rate_Evaluation_Data)
%     ylim([-max(Marginal_Emissions_Rate_Evaluation_Data) * 0.1, max(Marginal_Emissions_Rate_Evaluation_Data) * 1.1]) % Make ylim 10% larger than emissions rate range.
%     ylabel('Marginal Emissions Rate (metric tons/kWh)','FontSize',15);
%     title('Electricity Rates and Marginal Emissions Rates','FontSize',15)
%     legend('Electricity Rates ($/kWh)','Marginal Carbon Emissions Rate (metric tons/kWh)', ...
%         'Location','NorthOutside')
%     set(gca,'FontSize',15);
    
    % Plot Volumetric Rate Data without Marginal Emissions Rate
    figure('NumberTitle', 'off')
    plot(t, Volumetric_Rate_Data)
    xlim([t(1), t(end)])
    ylim([-max(Volumetric_Rate_Data) * 0.1, max(Volumetric_Rate_Data) * 1.1]) % Make ylim 10% larger than volumetric rate range.
    xlabel('Date & Time','FontSize',15);
    ylabel('Total Energy Charges ($/kWh)', 'FontSize', 15);
    title('Total Energy Charges','FontSize',15)
    set(gca,'FontSize',15);

    
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Energy Price and Carbon Plot.png");
        
        saveas(gcf, Output_Directory_Filepath + "Energy Price and Carbon Plot");
        
    end
    
end

%% Plot Coincident and Non-Coincident Demand Charge Schedule

% Create Summer/Winter Binary Flag Vector
Summer_Binary_Data = sum(Month_Data == First_Summer_Month:Last_Summer_Month, 2);

Winter_Binary_Data = sum(Month_Data == [1:(First_Summer_Month-1), (Last_Summer_Month+1):12], 2);

% Create Total-Demand-Charge Vector
% Noncoincident Demand Charge is always included (although it may be 0).
% Coincident Peak and Part-Peak values are only added if they are non-zero
% and a binary-flag data input is available.

Total_DC = (Winter_Noncoincident_DC * Winter_Binary_Data) + ...
    (Summer_Noncoincident_DC * Summer_Binary_Data);

if Winter_Special_Maximum_DC > 0
    Total_DC = Total_DC + (Winter_Special_Maximum_DC * Special_Maximum_Demand_Binary_Data);
end

if Winter_Peak_DC > 0
    Total_DC = Total_DC + (Winter_Peak_DC * Winter_Peak_Binary_Data);
end

if Winter_Part_Peak_DC > 0
    Total_DC = Total_DC + (Winter_Part_Peak_DC * Winter_Part_Peak_Binary_Data);
end

if Summer_Special_Maximum_DC > 0
    Total_DC = Total_DC + (Summer_Special_Maximum_DC * Special_Maximum_Demand_Binary_Data);
end

if Summer_Peak_DC > 0
    Total_DC = Total_DC + (Summer_Peak_DC * Summer_Peak_Binary_Data);
end

if Summer_Part_Peak_DC > 0
    Total_DC = Total_DC + (Summer_Part_Peak_DC * Summer_Part_Peak_Binary_Data);
end


if Show_Plots == 1 || Export_Plots ==1

    figure('NumberTitle', 'off')
    plot(t, Total_DC,'Color',[0,0.5,0])
    xlim([t(1), t(end)])
    ylim([-1, max(Total_DC) + 1])
    xlabel('Date & Time','FontSize',15);
    ylabel('Total Demand Charges ($/kW)','FontSize',15);
    title('Total Demand Charges','FontSize',15)
        
    
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Demand Charge Plot.png");
        
        saveas(gcf, Output_Directory_Filepath + "Demand Charge Plot");
        
    end
    
end


%% Plot Load, Net Load with Solar Only, Net Load with Solar and Storage

if Show_Plots == 1 || Export_Plots ==1
    
    if Model_Type_Input == "Storage Only"
        
        figure('NumberTitle', 'off')
        plot(t, Load_Profile_Data,'k', ...
            t, Load_Profile_Data - P_ES,'r')
        xlim([t(1), t(end)])
        xlabel('Date & Time','FontSize',15);
        ylabel('Load (kW)','FontSize',15);
        title('Original and Net Load Profiles','FontSize',15)
        legend('Original Load', 'Net Load with Storage', 'Location','NorthOutside')
        set(gca,'FontSize',15);
        
    elseif Model_Type_Input == "Solar Plus Storage"
        
        figure('NumberTitle', 'off')
        plot(t, Load_Profile_Data,'k', ...
            t, Load_Profile_Data - Solar_PV_Profile_Data,'b', ...
            t, Load_Profile_Data - (Solar_PV_Profile_Data + P_ES),'r')
        xlim([t(1), t(end)])
        xlabel('Date & Time','FontSize',15);
        ylabel('Load (kW)','FontSize',15);
        title('Original and Net Load Profiles','FontSize',15)
        legend('Original Load','Net Load with Solar Only', 'Net Load with Solar + Storage', 'Location','NorthOutside')
        set(gca,'FontSize',15);
        
    end
        
    
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Net Load Plot.png");
        
        saveas(gcf, Output_Directory_Filepath + "Net Load Plot");
        
    end
    
    
end

if Model_Type_Input == "Storage Only"
    
    Annual_Peak_Demand_with_Solar_Only = "";
    
    Annual_Total_Energy_Consumption_with_Solar_Only = "";
    
elseif Model_Type_Input == "Solar Plus Storage"
    
    Annual_Peak_Demand_with_Solar_Only = max(Load_Profile_Data - Solar_PV_Profile_Data);
    
    Annual_Total_Energy_Consumption_with_Solar_Only = sum(Load_Profile_Data - Solar_PV_Profile_Data) * delta_t;
    
end

Annual_Peak_Demand_with_Solar_and_Storage = max(Load_Profile_Data - (Solar_PV_Profile_Data + P_ES));

Annual_Total_Energy_Consumption_with_Solar_and_Storage = sum(Load_Profile_Data - (Solar_PV_Profile_Data + P_ES)) * delta_t;

if Model_Type_Input == "Storage Only"
    Solar_Only_Peak_Demand_Reduction_Percentage = "";
    
elseif Model_Type_Input == "Solar Plus Storage"
    Solar_Only_Peak_Demand_Reduction_Percentage = ...
        ((Annual_Peak_Demand_Baseline - Annual_Peak_Demand_with_Solar_Only)/...
        Annual_Peak_Demand_Baseline) * 100;
end

Solar_Storage_Peak_Demand_Reduction_Percentage = ...
    ((Annual_Peak_Demand_Baseline - Annual_Peak_Demand_with_Solar_and_Storage)/...
    Annual_Peak_Demand_Baseline) * 100;

if Model_Type_Input == "Storage Only"
    Solar_Only_Energy_Consumption_Decrease_Percentage = "";
    
elseif Model_Type_Input == "Solar Plus Storage"
    Solar_Only_Energy_Consumption_Decrease_Percentage = ...
        ((Annual_Total_Energy_Consumption_Baseline - ...
        Annual_Total_Energy_Consumption_with_Solar_Only)/...
        Annual_Total_Energy_Consumption_Baseline) * 100;
end

Solar_Storage_Energy_Consumption_Decrease_Percentage = ...
    ((Annual_Total_Energy_Consumption_Baseline - ...
    Annual_Total_Energy_Consumption_with_Solar_and_Storage)/...
    Annual_Total_Energy_Consumption_Baseline) * 100;

sprintf('Baseline annual peak noncoincident demand is %0.00f kW.', ...
    Annual_Peak_Demand_Baseline)

if Model_Type_Input == "Storage Only"
    
    if Solar_Storage_Peak_Demand_Reduction_Percentage >= 0
        
        sprintf('Peak demand with storage is %0.00f kW, representing a DECREASE OF %0.02f%%.', ...
            Annual_Peak_Demand_with_Solar_and_Storage, Solar_Storage_Peak_Demand_Reduction_Percentage)
        
    elseif Solar_Storage_Peak_Demand_Reduction_Percentage < 0
        
        sprintf('Peak demand with storage is %0.00f kW, representing an INCREASE OF %0.02f%%.', ...
            Annual_Peak_Demand_with_Solar_and_Storage, -Solar_Storage_Peak_Demand_Reduction_Percentage)
        
    end        
    
    sprintf('Baseline annual total electricity consumption is %0.00f kWh.', ...
    Annual_Total_Energy_Consumption_Baseline)
    
    sprintf('Electricity consumption with storage is %0.00f kWh, representing an INCREASE OF %0.02f%%.', ...
        Annual_Total_Energy_Consumption_with_Solar_and_Storage, -Solar_Storage_Energy_Consumption_Decrease_Percentage)
    
elseif Model_Type_Input == "Solar Plus Storage"
    
    sprintf('Peak demand with solar only is %0.00f kW, representing a DECREASE OF %0.02f%%.', ...
        Annual_Peak_Demand_with_Solar_Only, Solar_Only_Peak_Demand_Reduction_Percentage)
    
    if Solar_Storage_Peak_Demand_Reduction_Percentage >= 0
        sprintf('Peak demand with solar and storage is %0.00f kW, representing a DECREASE OF %0.02f%%.', ...
            Annual_Peak_Demand_with_Solar_and_Storage, Solar_Storage_Peak_Demand_Reduction_Percentage)
        
    elseif Solar_Storage_Peak_Demand_Reduction_Percentage < 0
        sprintf('Peak demand with solar and storage is %0.00f kW, representing an INCREASE OF %0.02f%%.', ...
            Annual_Peak_Demand_with_Solar_and_Storage, -Solar_Storage_Peak_Demand_Reduction_Percentage)
        
    end
    
    sprintf('Baseline annual total electricity consumption is %0.00f kWh.', ...
    Annual_Total_Energy_Consumption_Baseline)
    
    sprintf('Electricity consumption with solar only is %0.00f kWh, representing a DECREASE OF %0.02f%%.', ...
        Annual_Total_Energy_Consumption_with_Solar_Only, Solar_Only_Energy_Consumption_Decrease_Percentage)
    
    sprintf('Electricity consumption with solar and storage is %0.00f kWh, representing a DECREASE OF %0.02f%%.', ...
        Annual_Total_Energy_Consumption_with_Solar_and_Storage, Solar_Storage_Energy_Consumption_Decrease_Percentage)
    
end


%% Plot Monthly Costs as Bar Plot

% Calculate Baseline Monthly Costs

Monthly_Costs_Matrix_Baseline = [Fixed_Charge_Vector, NC_DC_Baseline_Vector, SM_DC_Baseline_Vector,...
    CPK_DC_Baseline_Vector, CPP_DC_Baseline_Vector, Energy_Charge_Baseline_Vector];

Annual_Costs_Vector_Baseline = [sum(Fixed_Charge_Vector); ...
    sum(NC_DC_Baseline_Vector) + sum(SM_DC_Baseline_Vector) + sum(CPK_DC_Baseline_Vector) + sum(CPP_DC_Baseline_Vector);...
    sum(Energy_Charge_Baseline_Vector)];

Annual_Demand_Charge_Cost_Baseline = Annual_Costs_Vector_Baseline(2);
Annual_Energy_Charge_Cost_Baseline = Annual_Costs_Vector_Baseline(3);


% Calculate Monthly Costs With Solar Only

Monthly_Costs_Matrix_with_Solar_Only = [Fixed_Charge_Vector, NC_DC_with_Solar_Only_Vector, SM_DC_with_Solar_Only_Vector, ...
    CPK_DC_with_Solar_Only_Vector, CPP_DC_with_Solar_Only_Vector, Energy_Charge_with_Solar_Only_Vector];

Annual_Costs_Vector_with_Solar_Only = [sum(Fixed_Charge_Vector); ...
    sum(NC_DC_with_Solar_Only_Vector) + sum(SM_DC_with_Solar_Only_Vector) + sum(CPK_DC_with_Solar_Only_Vector) + sum(CPP_DC_with_Solar_Only_Vector);...
    sum(Energy_Charge_with_Solar_Only_Vector)];

if Model_Type_Input == "Storage Only"
    Annual_Demand_Charge_Cost_with_Solar_Only = "";
    Annual_Energy_Charge_Cost_with_Solar_Only = "";
    
elseif Model_Type_Input == "Solar Plus Storage"
    Annual_Demand_Charge_Cost_with_Solar_Only = Annual_Costs_Vector_with_Solar_Only(2);
    Annual_Energy_Charge_Cost_with_Solar_Only = Annual_Costs_Vector_with_Solar_Only(3);
end


% Calculate Monthly Costs with Solar and Storage

Monthly_Costs_Matrix_with_Solar_and_Storage = [Fixed_Charge_Vector, NC_DC_with_Solar_and_Storage_Vector, SM_DC_with_Solar_and_Storage_Vector, ...
    CPK_DC_with_Solar_and_Storage_Vector, CPP_DC_with_Solar_and_Storage_Vector, Energy_Charge_with_Solar_and_Storage_Vector];

Annual_Costs_Vector_with_Solar_and_Storage = [sum(Fixed_Charge_Vector); ...
    sum(NC_DC_with_Solar_and_Storage_Vector) + sum(SM_DC_with_Solar_and_Storage_Vector) + sum(CPK_DC_with_Solar_and_Storage_Vector) + sum(CPP_DC_with_Solar_and_Storage_Vector);...
    sum(Energy_Charge_with_Solar_and_Storage_Vector)];

Annual_Demand_Charge_Cost_with_Solar_and_Storage = Annual_Costs_Vector_with_Solar_and_Storage(2);
Annual_Energy_Charge_Cost_with_Solar_and_Storage = Annual_Costs_Vector_with_Solar_and_Storage(3);


% Calculate Maximum and Minimum Monthly Bills - to set y-axis for all plots

Maximum_Monthly_Bill_Baseline = max(sum(Monthly_Costs_Matrix_Baseline, 2));
Minimum_Monthly_Bill_Baseline = min(sum(Monthly_Costs_Matrix_Baseline, 2));

Maximum_Monthly_Bill_with_Solar_Only = max(sum(Monthly_Costs_Matrix_with_Solar_Only, 2));
Minimum_Monthly_Bill_with_Solar_Only = min(sum(Monthly_Costs_Matrix_with_Solar_Only, 2));

Maximum_Monthly_Bill_with_Solar_and_Storage = max(sum(Monthly_Costs_Matrix_with_Solar_and_Storage, 2));
Minimum_Monthly_Bill_with_Solar_and_Storage = min(sum(Monthly_Costs_Matrix_with_Solar_and_Storage, 2));

Maximum_Monthly_Bill = max([Maximum_Monthly_Bill_Baseline, ...
    Maximum_Monthly_Bill_with_Solar_Only, ...
    Maximum_Monthly_Bill_with_Solar_and_Storage]);

Minimum_Monthly_Bill = min([Minimum_Monthly_Bill_Baseline, ...
    Minimum_Monthly_Bill_with_Solar_Only, ...
    Minimum_Monthly_Bill_with_Solar_and_Storage]);

Max_Monthly_Bill_ylim = Maximum_Monthly_Bill * 1.1; % Make upper ylim 10% larger than largest monthly bill.

if Minimum_Monthly_Bill >= 0
    Min_Monthly_Bill_ylim = 0; % Make lower ylim equal to 0 if the lowest monthly bill is greater than zero.
elseif Minimum_Monthly_Bill < 0
    Min_Monthly_Bill_ylim = Minimum_Monthly_Bill * 1.1; % Make lower ylim 10% smaller than the smallest monthly bill if less than zero.
end


% Plot Baseline Monthly Costs

if Show_Plots == 1 || Export_Plots ==1
    
    figure('NumberTitle', 'off')
    bar(Monthly_Costs_Matrix_Baseline, 'stacked')
    xlim([0.5, 12.5])
    ylim([Min_Monthly_Bill_ylim, Max_Monthly_Bill_ylim])
    xlabel('Month','FontSize',15);
    ylabel('Cost ($/Month)','FontSize',15);
    title('Monthly Costs, Without Storage','FontSize',15)
    legend('Fixed Charges','Max DC', 'Special Max DC', 'Peak DC','Part-Peak DC', 'Energy Charge', ...
        'Location', 'NorthWest')
    set(gca,'FontSize',15);
        
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Monthly Costs Baseline Plot.png");
        
        saveas(gcf, Output_Directory_Filepath + "Monthly Costs Baseline Plot");
        
    end
    
end


% Plot Monthly Costs With Solar Only

if Model_Type_Input == "Solar Plus Storage"
   
    if Show_Plots == 1 || Export_Plots ==1
        
        figure('NumberTitle', 'off')
        bar(Monthly_Costs_Matrix_with_Solar_Only, 'stacked')
        xlim([0.5, 12.5])
        ylim([Min_Monthly_Bill_ylim, Max_Monthly_Bill_ylim])
        xlabel('Month','FontSize',15);
        ylabel('Cost ($/Month)','FontSize',15);
        title('Monthly Costs, With Solar Only','FontSize',15)
        legend('Fixed Charges','Max DC', 'Special Max DC', 'Peak DC','Part-Peak DC', 'Energy Charge', ...
            'Location', 'NorthWest')
        set(gca,'FontSize',15);

        
        if Export_Plots == 1
            
            saveas(gcf, Output_Directory_Filepath + "Monthly Costs with Solar Only Plot.png");
            
            saveas(gcf, Output_Directory_Filepath + "Monthly Costs with Solar Only Plot");
            
        end
        
    end

end


% Plot Monthly Costs with Solar and Storage

if Show_Plots == 1 || Export_Plots ==1
    
    figure('NumberTitle', 'off')
    bar(Monthly_Costs_Matrix_with_Solar_and_Storage, 'stacked')
    xlim([0.5, 12.5])
    ylim([Min_Monthly_Bill_ylim, Max_Monthly_Bill_ylim])
    xlabel('Month','FontSize',15);
    ylabel('Cost ($/Month)','FontSize',15);
    title('Monthly Costs, With Storage','FontSize',15)
    legend('Fixed Charges','Max DC', 'Special Max DC', 'Peak DC','Part-Peak DC', 'Energy Charge', ...
        'Location', 'NorthWest')
    set(gca,'FontSize',15);
        
    
    if Export_Plots == 1
        
        if Model_Type_Input == "Storage Only"
            
            saveas(gcf, Output_Directory_Filepath + "Monthly Costs with Storage Plot.png");
            saveas(gcf, Output_Directory_Filepath + "Monthly Costs with Storage Plot");
            
        elseif Model_Type_Input == "Solar Plus Storage"
            
            saveas(gcf, Output_Directory_Filepath + "Monthly Costs with Solar and Storage Plot.png");
            saveas(gcf, Output_Directory_Filepath + "Monthly Costs with Solar and Storage Plot");
            
        end
        
    end
    
end


% Plot Monthly Savings From Storage

if Model_Type_Input == "Storage Only"
    
    Monthly_Savings_Matrix_From_Storage = Monthly_Costs_Matrix_Baseline - ...
        Monthly_Costs_Matrix_with_Solar_and_Storage;
    
elseif Model_Type_Input == "Solar Plus Storage"
    
    Monthly_Savings_Matrix_From_Storage = Monthly_Costs_Matrix_with_Solar_Only - ...
        Monthly_Costs_Matrix_with_Solar_and_Storage;
    
end


% Remove fixed charges, battery cycling costs.
Monthly_Savings_Matrix_Plot = Monthly_Savings_Matrix_From_Storage(:, 2:6);

if Show_Plots == 1 || Export_Plots ==1
    
    figure('NumberTitle', 'off')
    bar(Monthly_Savings_Matrix_Plot, 'stacked')
    xlim([0.5, 12.5])
    xlabel('Month','FontSize',15);
    xticks(linspace(1,12,12));
    ylabel('Savings ($/Month)','FontSize',15);
    title('Monthly Savings From Storage','FontSize',15)
    legend('Max DC', 'Special Max DC', 'Peak DC','Part-Peak DC', 'Energy Charge', ...
        'Location', 'NorthWest')
    set(gca,'FontSize',15);
        
    
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Monthly Savings from Storage Plot.png");
        
        saveas(gcf, Output_Directory_Filepath + "Monthly Savings from Storage Plot");
        
    end
    
    
end


%% Report Annual Savings

% Report Baseline Cost without Solar or Storage
Annual_Customer_Bill_Baseline = sum(sum(Monthly_Costs_Matrix_Baseline));

if Model_Type_Input == "Storage Only"
    Annual_Customer_Bill_with_Solar_Only = "";
    
elseif Model_Type_Input == "Solar Plus Storage"
    Annual_Customer_Bill_with_Solar_Only = sum(Annual_Costs_Vector_with_Solar_Only);
end

Annual_Customer_Bill_with_Solar_and_Storage = sum(Annual_Costs_Vector_with_Solar_and_Storage); % Doesn't include degradation cost.

if Model_Type_Input == "Storage Only"
    
    Annual_Customer_Bill_Savings_from_Storage = Annual_Customer_Bill_Baseline - Annual_Customer_Bill_with_Solar_and_Storage;
    
elseif Model_Type_Input == "Solar Plus Storage"
    
    Annual_Customer_Bill_Savings_from_Solar = Annual_Customer_Bill_Baseline - Annual_Customer_Bill_with_Solar_Only;
    
    Annual_Customer_Bill_Savings_from_Solar_Percent = (Annual_Customer_Bill_Savings_from_Solar/Annual_Customer_Bill_Baseline);
    
    Annual_Customer_Bill_Savings_from_Storage = Annual_Customer_Bill_with_Solar_Only - Annual_Customer_Bill_with_Solar_and_Storage;
    
end

Annual_Customer_Bill_Savings_from_Storage_Percent = (Annual_Customer_Bill_Savings_from_Storage/Annual_Customer_Bill_Baseline);

if Model_Type_Input == "Solar Plus Storage"
    Solar_Installed_Cost = Solar_Size_Input * Solar_Installed_Cost_per_kW;
    Solar_Simple_Payback = Solar_Installed_Cost/Annual_Customer_Bill_Savings_from_Solar;
    
    sprintf('Annual cost savings from solar is $%0.0f, representing %0.2f%% of the original $%0.0f bill.', ...
        Annual_Customer_Bill_Savings_from_Solar, Annual_Customer_Bill_Savings_from_Solar_Percent * 100, Annual_Customer_Bill_Baseline)
    
    sprintf('The solar PV system has a simple payback of %0.0f years, not including incentives.', ...
        Solar_Simple_Payback)
end

Storage_Installed_Cost = Total_Storage_Capacity * Storage_Installed_Cost_per_kWh;

Storage_Simple_Payback = Storage_Installed_Cost/Annual_Customer_Bill_Savings_from_Storage;

sprintf('Annual cost savings from storage is $%0.0f, representing %0.2f%% of the original $%0.0f bill.', ...
    Annual_Customer_Bill_Savings_from_Storage, Annual_Customer_Bill_Savings_from_Storage_Percent * 100, Annual_Customer_Bill_Baseline)

sprintf('The storage system has a simple payback of %0.0f years, not including incentives.', ...
    Storage_Simple_Payback)


%% Report Cycling/Degradation Penalty

Annual_Equivalent_Storage_Cycles = sum(Cycles_Vector);
Annual_Cycling_Penalty = sum(Cycling_Penalty_Vector);
Annual_Capacity_Fade = Usable_Storage_Capacity_Input - Usable_Storage_Capacity;
sprintf('The battery cycles %0.0f times annually, with a degradation cost of $%0.0f, and experiences capacity fade of %0.1f kWh.', ...
    Annual_Equivalent_Storage_Cycles, Annual_Cycling_Penalty, Annual_Capacity_Fade)

%% Report Operational/"SGIP" Round-Trip Efficiency

Annual_RTE = (sum(P_ES_out) * delta_t)/(sum(P_ES_in) * delta_t);

sprintf('The battery has an Annual Operational/SGIP Round-Trip Efficiency of %0.2f%%.', ...
    Annual_RTE * 100)


%% Report Operational/"SGIP" Capacity Factor

% The SGIP Handbook uses the following definition of capacity factor for
% storage resources, based on the assumption that 60% of hours are
% available for discharge. The term "hours of data available" is equal to
% the number of hours in the year here. For actual operational data, it's
% the number of hours where data is available, which may be less than the
% number of hours in the year. Here, the number of hours in the year is
% calculated by multiplying the number of timesteps of original load profile data
% by the timestep length delta_t. This returns 8760 hours during
% non-leap years and 8784 during leap years.

% Capacity Factor = (kWh Discharge)/(Hours of Data Available x Rebated Capacity (kW) x 60%)

Operational_Capacity_Factor = ((sum(P_ES_out) * delta_t)/((length(Load_Profile_Data) * delta_t) * Storage_Power_Rating_Input * 0.6));

sprintf('The battery has an Operational/SGIP Capacity Factor of %0.2f%%.', ...
    Operational_Capacity_Factor * 100)


%% Report Grid Costs

% Calculate Total Annual Grid Costs

Annual_Grid_Cost_Baseline = (Generation_Cost_Data + Representative_Distribution_Cost_Data)' * ...
    Load_Profile_Data * delta_t;

if Model_Type_Input == "Solar Plus Storage"
    Annual_Grid_Cost_with_Solar_Only = (Generation_Cost_Data + Representative_Distribution_Cost_Data)' * ...
        (Load_Profile_Data - Solar_PV_Profile_Data) * delta_t;
else
    Annual_Grid_Cost_with_Solar_Only = "";
end

Annual_Grid_Cost_with_Solar_and_Storage = (Generation_Cost_Data + Representative_Distribution_Cost_Data)' * ...
    (Load_Profile_Data - Solar_PV_Profile_Data - P_ES_out + P_ES_in) * delta_t;


% Calculate Monthly Grid Costs

Grid_Cost_Timestep_Baseline = [Generation_Cost_Data .* Load_Profile_Data * delta_t, ...
    Representative_Distribution_Cost_Data .* Load_Profile_Data * delta_t];

Grid_Cost_Month_Baseline = [];

for Month_Iter = 1:12
    Grid_Cost_Single_Month_Baseline = sum(Grid_Cost_Timestep_Baseline(Month_Data == Month_Iter, :));
    Grid_Cost_Month_Baseline = [Grid_Cost_Month_Baseline; Grid_Cost_Single_Month_Baseline];
end


Grid_Cost_Timestep_with_Solar_Only = [Generation_Cost_Data .* (Load_Profile_Data - Solar_PV_Profile_Data) * delta_t, ...
    Representative_Distribution_Cost_Data .* (Load_Profile_Data - Solar_PV_Profile_Data) * delta_t];

Grid_Cost_Month_with_Solar_Only = [];

for Month_Iter = 1:12
    Grid_Cost_Single_Month_with_Solar_Only = sum(Grid_Cost_Timestep_with_Solar_Only(Month_Data == Month_Iter, :));
    Grid_Cost_Month_with_Solar_Only = [Grid_Cost_Month_with_Solar_Only; Grid_Cost_Single_Month_with_Solar_Only];
end


Grid_Cost_Timestep_with_Solar_and_Storage = [Generation_Cost_Data .* (Load_Profile_Data - Solar_PV_Profile_Data - P_ES_out + P_ES_in) * delta_t, ...
    Representative_Distribution_Cost_Data .* (Load_Profile_Data - Solar_PV_Profile_Data - P_ES_out + P_ES_in) * delta_t];

Grid_Cost_Month_with_Solar_and_Storage = [];

for Month_Iter = 1:12
    Grid_Cost_Single_Month_with_Solar_and_Storage = sum(Grid_Cost_Timestep_with_Solar_and_Storage(Month_Data == Month_Iter, :));
    Grid_Cost_Month_with_Solar_and_Storage = [Grid_Cost_Month_with_Solar_and_Storage; Grid_Cost_Single_Month_with_Solar_and_Storage];
end


% Calculate Monthly Grid Cost Savings from Storage

if Model_Type_Input == "Storage Only"
    
    Grid_Cost_Savings_Month_from_Storage = Grid_Cost_Month_Baseline - Grid_Cost_Month_with_Solar_and_Storage;
    
elseif Model_Type_Input == "Solar Plus Storage"
    
    Grid_Cost_Savings_Month_from_Storage = Grid_Cost_Month_with_Solar_Only - Grid_Cost_Month_with_Solar_and_Storage;
    
end


% Report Grid Cost Savings from Solar

if Model_Type_Input == "Solar Plus Storage"
    
    sprintf('Installing solar DECREASES estimated utility grid costs (not including transmission costs, \n and using representative distribution costs) by $%0.2f per year.', ...
        Annual_Grid_Cost_Baseline - Annual_Grid_Cost_with_Solar_Only)   

end


% Report Grid Cost Impact from Storage

if Model_Type_Input == "Storage Only"
    
    if Annual_Grid_Cost_Baseline - Annual_Grid_Cost_with_Solar_and_Storage < 0
        sprintf('Installing energy storage INCREASES estimated utility grid costs (not including transmission costs, \n and using representative distribution costs) by $%0.2f per year.', ...
            -(Annual_Grid_Cost_Baseline - Annual_Grid_Cost_with_Solar_and_Storage))
    else
        sprintf('Installing energy storage DECREASES estimated utility grid costs (not including transmission costs, \n and using representative distribution costs) by $%0.2f per year.', ...
            Annual_Grid_Cost_Baseline - Annual_Grid_Cost_with_Solar_and_Storage)
    end
    
elseif Model_Type_Input == "Solar Plus Storage"
    
    if Annual_Grid_Cost_with_Solar_Only - Annual_Grid_Cost_with_Solar_and_Storage < 0
        sprintf('Installing energy storage INCREASES estimated utility grid costs (not including transmission costs, \n and using representative distribution costs) by $%0.2f per year.', ...
            -(Annual_Grid_Cost_with_Solar_Only - Annual_Grid_Cost_with_Solar_and_Storage))
    else
        sprintf('Installing energy storage DECREASES estimated utility grid costs (not including transmission costs, \n and using representative distribution costs) by $%0.2f per year.', ...
            Annual_Grid_Cost_with_Solar_Only - Annual_Grid_Cost_with_Solar_and_Storage)
    end
    
end

%% Report Emissions Impact

% This approach multiplies net load by marginal emissions factors to
% calculate total annual emissions. This is consistent with the idea that
% the customer would pay an adder based on marginal emissions factors.
% Typically, total annual emissions is calculated using average emissions
% values, not marginal emissions values.

% https://www.pge.com/includes/docs/pdfs/shared/environment/calculator/pge_ghg_emission_factor_info_sheet.pdf

% (tons/kWh) = (tons/MWh) * (MWh/kWh)
Annual_GHG_Emissions_Baseline = (Marginal_Emissions_Rate_Evaluation_Data' * Load_Profile_Data * ...
    (1/1000) * delta_t);

if Model_Type_Input == "Storage Only"
    Annual_GHG_Emissions_with_Solar_Only = "";
    
elseif Model_Type_Input == "Solar Plus Storage"
    Annual_GHG_Emissions_with_Solar_Only = (Marginal_Emissions_Rate_Evaluation_Data' * (Load_Profile_Data - Solar_PV_Profile_Data) * ...
        (1/1000) * delta_t);
end

Annual_GHG_Emissions_with_Solar_and_Storage = (Marginal_Emissions_Rate_Evaluation_Data' * (Load_Profile_Data - ...
    (Solar_PV_Profile_Data + P_ES_out - P_ES_in)) * (1/1000) * delta_t);

if Model_Type_Input == "Storage Only"
    Annual_GHG_Emissions_Reduction_from_Solar = "";
elseif Model_Type_Input == "Solar Plus Storage"
    Annual_GHG_Emissions_Reduction_from_Solar = Annual_GHG_Emissions_Baseline - Annual_GHG_Emissions_with_Solar_Only;
end

if Model_Type_Input == "Storage Only"
    Annual_GHG_Emissions_Reduction_from_Storage = Annual_GHG_Emissions_Baseline - Annual_GHG_Emissions_with_Solar_and_Storage;
elseif Model_Type_Input == "Solar Plus Storage"
    Annual_GHG_Emissions_Reduction_from_Storage = Annual_GHG_Emissions_with_Solar_Only - Annual_GHG_Emissions_with_Solar_and_Storage;
end

if Model_Type_Input == "Storage Only"
    Annual_GHG_Emissions_Reduction_from_Solar_Percent = "";
elseif Model_Type_Input == "Solar Plus Storage"
    Annual_GHG_Emissions_Reduction_from_Solar_Percent = ...
        (Annual_GHG_Emissions_Reduction_from_Solar/Annual_GHG_Emissions_Baseline);
end

Annual_GHG_Emissions_Reduction_from_Storage_Percent = ...
    (Annual_GHG_Emissions_Reduction_from_Storage/Annual_GHG_Emissions_Baseline);


if Model_Type_Input == "Solar Plus Storage"
    
    sprintf('Installing solar DECREASES marginal carbon emissions \n by %0.2f metric tons per year.', ...
        Annual_GHG_Emissions_Reduction_from_Solar)
    sprintf('This is equivalent to %0.2f%% of baseline emissions, and brings total emissions to %0.2f metric tons per year.', ...
        Annual_GHG_Emissions_Reduction_from_Solar_Percent * 100, Annual_GHG_Emissions_with_Solar_Only)
    
end


if Annual_GHG_Emissions_Reduction_from_Storage < 0
    sprintf('Installing energy storage INCREASES marginal carbon emissions \n by %0.2f metric tons per year.', ...
        -Annual_GHG_Emissions_Reduction_from_Storage)
    sprintf('This is equivalent to %0.2f%% of baseline emissions, and brings total emissions to %0.2f metric tons per year.', ...
        -Annual_GHG_Emissions_Reduction_from_Storage_Percent * 100, Annual_GHG_Emissions_with_Solar_and_Storage)
else
    sprintf('Installing energy storage DECREASES marginal carbon emissions \n by %0.2f metric tons per year.', ...
        Annual_GHG_Emissions_Reduction_from_Storage)
    sprintf('This is equivalent to %0.2f%% of baseline emissions, and brings total emissions to %0.2f metric tons per year.', ...
        Annual_GHG_Emissions_Reduction_from_Storage_Percent * 100, Annual_GHG_Emissions_with_Solar_and_Storage)
end


%% Plot Grid Costs

% Plot Grid Cost Time-Series

if Show_Plots == 1 || Export_Plots ==1
    
    figure('NumberTitle', 'off')
    plot(t, Generation_Cost_Data, ...
        t, Representative_Distribution_Cost_Data)
    xlim([t(1), t(end)])
    xlabel('Date & Time','FontSize',15);
    ylim([-max([Generation_Cost_Data; Representative_Distribution_Cost_Data]) * 0.1, ...
        max([Generation_Cost_Data; Representative_Distribution_Cost_Data]) * 1.1]) % Make ylim 10% larger than grid cost range.
    ylabel('Grid Costs ($/kWh)','FontSize',15);
    title('Grid Costs (Generation & Distribution)','FontSize',15)
    legend('Grid Generation Cost', 'Representative Distribution Cost', 'Location','NorthOutside')
    set(gca,'FontSize',15);
    
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Grid Costs Time Series Plot.png");
        
        saveas(gcf, Output_Directory_Filepath + "Grid Costs Time Series Plot");
        
    end
    
end
        

% Calculate Maximum and Minimum Monthly Grid Costs - to set y-axis for all plots

Maximum_Monthly_Grid_Cost_Baseline = max(sum(Grid_Cost_Month_Baseline, 2));
Minimum_Monthly_Grid_Cost_Baseline = min(sum(Grid_Cost_Month_Baseline, 2));

Grid_Cost_Month_with_Solar_Only_Neg = Grid_Cost_Month_with_Solar_Only;
Grid_Cost_Month_with_Solar_Only_Neg(Grid_Cost_Month_with_Solar_Only_Neg > 0) = 0;
Grid_Cost_Month_with_Solar_Only_Pos = Grid_Cost_Month_with_Solar_Only;
Grid_Cost_Month_with_Solar_Only_Pos(Grid_Cost_Month_with_Solar_Only_Pos < 0) = 0;

Maximum_Monthly_Grid_Cost_with_Solar_Only = max(sum(Grid_Cost_Month_with_Solar_Only_Pos, 2));
Minimum_Monthly_Grid_Cost_with_Solar_Only = min(sum(Grid_Cost_Month_with_Solar_Only_Neg, 2));

Grid_Cost_Month_with_Solar_and_Storage_Neg = Grid_Cost_Month_with_Solar_and_Storage;
Grid_Cost_Month_with_Solar_and_Storage_Neg(Grid_Cost_Month_with_Solar_and_Storage_Neg > 0) = 0;
Grid_Cost_Month_with_Solar_and_Storage_Pos = Grid_Cost_Month_with_Solar_and_Storage;
Grid_Cost_Month_with_Solar_and_Storage_Pos(Grid_Cost_Month_with_Solar_and_Storage_Pos < 0) = 0;

Maximum_Monthly_Grid_Cost_with_Solar_and_Storage = max(sum(Grid_Cost_Month_with_Solar_and_Storage_Pos, 2));
Minimum_Monthly_Grid_Cost_with_Solar_and_Storage = min(sum(Grid_Cost_Month_with_Solar_and_Storage_Neg, 2));

Maximum_Monthly_Grid_Cost = max([Maximum_Monthly_Grid_Cost_Baseline, ...
    Maximum_Monthly_Grid_Cost_with_Solar_Only, ...
    Maximum_Monthly_Grid_Cost_with_Solar_and_Storage]);

Minimum_Monthly_Grid_Cost = min([Minimum_Monthly_Grid_Cost_Baseline, ...
    Minimum_Monthly_Grid_Cost_with_Solar_Only, ...
    Minimum_Monthly_Grid_Cost_with_Solar_and_Storage]);

Max_Monthly_Grid_Cost_ylim = Maximum_Monthly_Grid_Cost * 1.1; % Make upper ylim 10% larger than largest monthly bill.

if Minimum_Monthly_Grid_Cost >= 0
    Min_Monthly_Grid_Cost_ylim = 0; % Make lower ylim equal to 0 if the lowest monthly bill is greater than zero.
elseif Minimum_Monthly_Grid_Cost < 0
    Min_Monthly_Grid_Cost_ylim = Minimum_Monthly_Grid_Cost * 1.1; % Make lower ylim 10% smaller than the smallest monthly bill if less than zero.
end


% Plot Baseline Monthly Grid Costs

if Show_Plots == 1 || Export_Plots ==1
    
    figure('NumberTitle', 'off')
    bar(Grid_Cost_Month_Baseline, 'stacked')
    xlim([0.5, 12.5])
    ylim([Min_Monthly_Grid_Cost_ylim, Max_Monthly_Grid_Cost_ylim])
    xlabel('Month','FontSize',15);
    xticks(linspace(1,12,12));
    ylabel('Grid Cost ($/month)','FontSize',15);
    title('Monthly Baseline Grid Costs','FontSize',15);
    legend('Generation Cost', 'Representative Distribution Cost', 'Location', 'NorthWest');
    set(gca,'FontSize',15);
    
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Monthly Grid Costs Baseline Plot.png");
        
        saveas(gcf, Output_Directory_Filepath + "Monthly Grid Costs Baseline Plot");
        
    end
    
end


% Plot Monthly Grid Costs With Solar Only

if Model_Type_Input == "Solar Plus Storage"
    
    if Show_Plots == 1 || Export_Plots ==1
               
        figure('NumberTitle', 'off')
        hold on
        bar(Grid_Cost_Month_with_Solar_Only_Neg, 'stacked')
        ax = gca;
        ax.ColorOrderIndex = 1; % Reset Color Order
        bar(Grid_Cost_Month_with_Solar_Only_Pos, 'stacked')
        hold off
        xlim([0.5, 12.5])
        ylim([Min_Monthly_Grid_Cost_ylim, Max_Monthly_Grid_Cost_ylim])
        xlabel('Month','FontSize',15);
        xticks(linspace(1,12,12));
        ylabel('Grid Cost ($/month)','FontSize',15);
        title('Monthly Grid Costs with Solar Only','FontSize',15);
        legend('Generation Cost', 'Representative Distribution Cost', 'Location', 'NorthWest');
        set(gca,'FontSize',15);
        
        if Export_Plots == 1
            
            saveas(gcf, Output_Directory_Filepath + "Monthly Grid Costs with Solar Only Plot.png");
            
            saveas(gcf, Output_Directory_Filepath + "Monthly Grid Costs with Solar Only Plot");
            
        end
        
    end
    
end


% Plot Monthly Grid Costs with Solar and Storage

if Show_Plots == 1 || Export_Plots ==1
        
    figure('NumberTitle', 'off')
    hold on
    bar(Grid_Cost_Month_with_Solar_and_Storage_Neg, 'stacked')
    ax = gca;
    ax.ColorOrderIndex = 1; % Reset Color Order
    bar(Grid_Cost_Month_with_Solar_and_Storage_Pos, 'stacked')
    hold off
    xlim([0.5, 12.5])
    ylim([Min_Monthly_Grid_Cost_ylim, Max_Monthly_Grid_Cost_ylim])
    xlabel('Month','FontSize',15);
    xticks(linspace(1,12,12));
    ylabel('Grid Cost ($/month)','FontSize',15);
    title('Monthly Grid Costs with Storage','FontSize',15);
    legend('Generation Cost', 'Representative Distribution Cost', 'Location', 'NorthWest');
    set(gca,'FontSize',15);
    
    
    if Export_Plots == 1
        
        if Model_Type_Input == "Storage Only"
            
            saveas(gcf, Output_Directory_Filepath + "Monthly Grid Costs with Storage Plot.png");
            saveas(gcf, Output_Directory_Filepath + "Monthly Grid Costs with Storage Plot");
            
        elseif Model_Type_Input == "Solar Plus Storage"
            
            saveas(gcf, Output_Directory_Filepath + "Monthly Grid Costs with Solar and Storage Plot.png");
            saveas(gcf, Output_Directory_Filepath + "Monthly Grid Costs with Solar and Storage Plot");
            
        end
        
    end
    
end


% Plot Monthly Savings from Storage

if Show_Plots == 1 || Export_Plots ==1
    
    % Separate negative and positive values for stacked bar chart
    Grid_Cost_Savings_Month_from_Storage_Neg = Grid_Cost_Savings_Month_from_Storage;
    Grid_Cost_Savings_Month_from_Storage_Neg(Grid_Cost_Savings_Month_from_Storage_Neg > 0) = 0;
    
    Grid_Cost_Savings_Month_from_Storage_Pos = Grid_Cost_Savings_Month_from_Storage;
    Grid_Cost_Savings_Month_from_Storage_Pos(Grid_Cost_Savings_Month_from_Storage_Pos < 0) = 0;
    
    
    % Calculate Maximum and Minimum Monthly Grid Savings - to set y-axis for plot
    
    Maximum_Grid_Cost_Savings_Month_from_Storage = max(sum(Grid_Cost_Savings_Month_from_Storage_Pos, 2));
    Minimum_Grid_Cost_Savings_Month_from_Storage = min(sum(Grid_Cost_Savings_Month_from_Storage_Neg, 2));
    
    Max_Grid_Cost_Savings_from_Storage_ylim = Maximum_Grid_Cost_Savings_Month_from_Storage * 1.1; % Make upper ylim 10% larger than largest monthly savings.
    
    if Minimum_Grid_Cost_Savings_Month_from_Storage >= 0
        Min_Grid_Cost_Savings_from_Storage_ylim = 0; % Make lower ylim equal to 0 if the lowest monthly grid savings.
    elseif Minimum_Grid_Cost_Savings_Month_from_Storage < 0
        Min_Grid_Cost_Savings_from_Storage_ylim = Minimum_Grid_Cost_Savings_Month_from_Storage * 1.1 - Max_Grid_Cost_Savings_from_Storage_ylim * 0.1; % Make lower ylim 10% smaller than the smallest monthly bill if less than zero.
    end
    
    
    figure('NumberTitle', 'off')
    hold on
    bar(Grid_Cost_Savings_Month_from_Storage_Neg, 'stacked')
    ax = gca;
    ax.ColorOrderIndex = 1; % Reset Color Order
    bar(Grid_Cost_Savings_Month_from_Storage_Pos, 'stacked')
    hold off
    xlim([0.5, 12.5])
    xlabel('Month','FontSize',15);
    xticks(linspace(1,12,12));
    ylim([Min_Grid_Cost_Savings_from_Storage_ylim, Max_Grid_Cost_Savings_from_Storage_ylim])
    ylabel('Grid Cost Savings ($/month)','FontSize',15);
    title('Monthly Grid Cost Savings from Storage','FontSize',15);
    legend('Generation Cost', 'Representative Distribution Cost', 'Location', 'NorthWest');
    set(gca,'FontSize',15);
    
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Monthly Grid Cost Savings from Storage Plot.png");
        saveas(gcf, Output_Directory_Filepath + "Monthly Grid Cost Savings from Storage Plot");
        
    end
    
end


%% Plot Emissions Impact by Month

if Show_Plots == 1 || Export_Plots ==1
    
    Emissions_Impact_Timestep = Marginal_Emissions_Rate_Evaluation_Data .* -P_ES * (1/1000) * delta_t;
    
    Emissions_Impact_Month = [];
    
    for Month_Iter = 1:12
        Emissions_Impact_Single_Month = sum(Emissions_Impact_Timestep(Month_Data == Month_Iter, :));
        Emissions_Impact_Month = [Emissions_Impact_Month; Emissions_Impact_Single_Month];
    end
    
    figure('NumberTitle', 'off')
    bar(Emissions_Impact_Month)
    xlim([0.5, 12.5])
    xlabel('Month','FontSize',15);
    xticks(linspace(1,12,12));
    ylabel('Emissions Increase (metric tons/month)','FontSize',15);
    title('Monthly Emissions Impact From Storage','FontSize',15)
    set(gca,'FontSize',15);
    
    hold on
    for i = 1:length(Emissions_Impact_Month)
        h=bar(i, Emissions_Impact_Month(i));
        if Emissions_Impact_Month(i) < 0
            set(h,'FaceColor',[0 0.5 0]);
        elseif Emissions_Impact_Month(i) >=0
            set(h,'FaceColor','r');
        end
    end
    hold off
    
    if Export_Plots == 1
        
        saveas(gcf, Output_Directory_Filepath + "Monthly Emissions Impact from Storage Plot.png");
        saveas(gcf, Output_Directory_Filepath + "Monthly Emissions Impact from Storage Plot");
        
    end
    
end
    
%% Close All Figures

if Show_Plots == 0
    
    close all;
    
end


%% Write Outputs to CSV

Model_Inputs_and_Outputs = table(Modeling_Team_Input, Model_Run_Number_Input, Model_Run_Date_Time, Model_Type_Input, ...
    Model_Timestep_Resolution, Customer_Class_Input, Load_Profile_Master_Index, ...
    Load_Profile_Name_Input, Retail_Rate_Master_Index, Retail_Rate_Utility, ...
    Retail_Rate_Name_Output, Retail_Rate_Effective_Date, ...
    Solar_Profile_Master_Index, Solar_Profile_Name_Output, Solar_Profile_Description, ...
    Solar_Size_Input, Storage_Type_Input, Storage_Power_Rating_Input, ...
    Usable_Storage_Capacity_Input, Single_Cycle_RTE_Input, Parasitic_Storage_Load_Input, ...
    Storage_Control_Algorithm_Name, Storage_Control_Algorithm_Description, ...
    Storage_Control_Algorithms_Parameters_Filename, ...  
    GHG_Reduction_Solution_Input, Equivalent_Cycling_Constraint_Input, ...
    Annual_RTE_Constraint_Input, ITC_Constraint_Input, ...
    Carbon_Adder_Incentive_Value_Input, Other_Incentives_or_Penalities, ...
    Emissions_Forecast_Signal_Input, ...
    Annual_GHG_Emissions_Baseline, Annual_GHG_Emissions_with_Solar_Only, ...
    Annual_GHG_Emissions_with_Solar_and_Storage, ...
    Annual_Customer_Bill_Baseline, Annual_Customer_Bill_with_Solar_Only, ...
    Annual_Customer_Bill_with_Solar_and_Storage, ...
    Annual_Grid_Cost_Baseline, Annual_Grid_Cost_with_Solar_Only, ...
    Annual_Grid_Cost_with_Solar_and_Storage, ...
    Annual_Equivalent_Storage_Cycles, Annual_RTE, Operational_Capacity_Factor, ...
    Annual_Demand_Charge_Cost_Baseline, Annual_Demand_Charge_Cost_with_Solar_Only, ...
    Annual_Demand_Charge_Cost_with_Solar_and_Storage, ...
    Annual_Energy_Charge_Cost_Baseline, Annual_Energy_Charge_Cost_with_Solar_Only, ...
    Annual_Energy_Charge_Cost_with_Solar_and_Storage, ...
    Annual_Peak_Demand_Baseline, Annual_Peak_Demand_with_Solar_Only, ...
    Annual_Peak_Demand_with_Solar_and_Storage, ...
    Annual_Total_Energy_Consumption_Baseline, Annual_Total_Energy_Consumption_with_Solar_Only, ...
    Annual_Total_Energy_Consumption_with_Solar_and_Storage, ...
    Output_Summary_Filename, Output_Description_Filename, Output_Visualizations_Filename, ...
    EV_Use, EV_Charge, EV_Gas_Savings, EV_GHG_Savings);

Storage_Dispatch_Outputs = table(t, P_ES);
Storage_Dispatch_Outputs.Properties.VariableNames = {'Date_Time_Pacific_No_DST', 'Storage_Output_kW'};

if Export_Data == 1
    
    writetable(Model_Inputs_and_Outputs, Output_Directory_Filepath + Output_Summary_Filename);
    
    writetable(Storage_Dispatch_Outputs, Output_Directory_Filepath + "Storage Dispatch Profile Output.csv");
    
end


%% Return to OSESMO Git Repository Directory

cd(OSESMO_Git_Repo_Directory)


end