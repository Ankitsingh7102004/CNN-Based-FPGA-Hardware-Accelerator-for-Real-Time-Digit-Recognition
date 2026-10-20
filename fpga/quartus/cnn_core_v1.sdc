# cnn_core_v1.sdc - timing constraints for the DE10-Standard MNIST CNN (laptop-camera version)

create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]
derive_clock_uncertainty

# KEY buttons are asynchronous and synchronized/debounced in RTL.
set_false_path -from [get_ports {KEY[*]}]
