#Column names of all columns with only 1 value
one_val_col = ['Filtered_Accelerator_Pedal_value', 'Inhibition_of_engine_fuel_cut_off', 'Fuel_Pressure', 
               'Torque_scaling_factor(standardization)', 'Glow_plug_control_request']

#Column names of numerically distributed columns
to_be_scaled_col = ['Fuel_consumption', 'Accelerator_Pedal_value', 'Throttle_position_signal', 'Short_Term_Fuel_Trim_Bank1', 
                    'Intake_air_pressure', 'Absolute_throttle_position', 'Engine_speed', 'Engine_torque_after_correction', 
                    'Torque_of_friction', 'Flywheel_torque_(after_torque_interventions)', 'Current_spark_timing', 
                    'Engine_coolant_temperature', 'Engine_Idel_Target_Speed', 'Engine_torque', 'Calculated_LOAD_value', 
                    'Maximum_indicated_engine_torque', 'Flywheel_torque', 'TCU_requests_engine_torque_limit_(ETL)', 
                    'TCU_requested_engine_RPM_increase', 'Torque_converter_speed', 'Engine_coolant_temperature.1', 
                    'Wheel_velocity_front_left-hand', 'Wheel_velocity_rear_right-hand', 'Wheel_velocity_front_right-hand', 
                    'Wheel_velocity_rear_left-hand', 'Torque_converter_turbine_speed_-_Unfiltered', 'Vehicle_speed', 
                    'Acceleration_speed_-_Longitudinal', 'Master_cylinder_pressure', 'Calculated_road_gradient', 
                    'Acceleration_speed_-_Lateral', 'Steering_wheel_speed', 'Steering_wheel_angle', 'Time(s)']

#Column names of columns with exactly two distinct values
two_val_col = ['Engine_in_fuel_cut_off', 'Standard_Torque_Ratio', 'Requested_spark_retard_angle_from_TCU', 
               'Target_engine_speed_used_in_lock-up_module', 'Activation_of_Air_compressor', 
               'Clutch_operation_acknowledge', 'PathOrder'] 

#Column names of columns with only a couple distinct numbers (like up to 10)
some_val_num_col = ['Engine_soacking_time', 'Long_Term_Fuel_Trim_Bank1', 'Minimum_indicated_engine_torque']

#Column names of columns with whole numbers, where numbers are more like categories
some_val_cat_col = ['Current_Gear', 'Converter_clutch', 'Gear_Selection', 'Indication_of_brake_switch_ON/OFF'] 
