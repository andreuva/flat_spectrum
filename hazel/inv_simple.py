import hazel

mod = hazel.Model('conf_test_nlte.ini', working_mode='inversion')
mod.read_observation()
mod.open_output()
mod.invert()
mod.write_output()
mod.close_output()

