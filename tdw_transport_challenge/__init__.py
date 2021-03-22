from gym.envs.registration import register

register(
    id='transport_challenge-v0',
    entry_point='tdw_transport_challenge.tdw_gym:TDW'
)