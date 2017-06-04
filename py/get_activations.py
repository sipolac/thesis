def get_params_appliance():
    '''Defines parameters for all appliances of interest'''

    # Credit: Mingjun Zhong and Jack Kelly
    # TODO: adjust max_on_power and other thresholds based on REFIT dataset.
    return {
        'kettle':{
            'windowlength':129,
            'on_power_threshold':2000,
            'max_on_power':3998,
            'mean':700,
            'std':1000,
            's2s_length':128},
        'microwave':{
            'windowlength':129,
            'on_power_threshold':200,
            'max_on_power':3969,
            'mean':500,
            'std':800,
            's2s_length':128},
        'fridge':{
            'windowlength':299,
            'on_power_threshold':50,
            'max_on_power':3323,
            'mean':200,
            'std':400,
            's2s_length':512},
        'dishwasher':{
            'windowlength':599,
            'on_power_threshold':10,
            'max_on_power':3964,
            'mean':700,
            'std':1000,
            's2s_length':1536},
        'washingmachine':{
            'windowlength':599,
            'on_power_threshold':20,
            'max_on_power':3999,
            'mean':400,
            'std':700,
            's2s_length':2000}
    }