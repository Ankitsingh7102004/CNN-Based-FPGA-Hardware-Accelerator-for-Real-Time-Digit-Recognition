# =============================================================================
# de10_setup.tcl - adds source files and DE10-Standard pin assignments.
# Pins taken from the DE10-Standard user-manual pin tables.
#
# Run it with your project open in Quartus:
#   Tools > Tcl Scripts... > select this file > Run
# (or in the Tcl console:  source de10_setup.tcl)
# =============================================================================

set_global_assignment -name TOP_LEVEL_ENTITY mnist_fpga_top
set_global_assignment -name VERILOG_FILE mnist_fpga_top.v
set_global_assignment -name VERILOG_FILE cnn_core_v1.v

set pins {
    CLOCK_50  PIN_AF14
    {KEY[0]}  PIN_AJ4
    {KEY[1]}  PIN_AK4
    {KEY[2]}  PIN_AA14
    {KEY[3]}  PIN_AA15
    {LEDR[0]} PIN_AA24
    {LEDR[1]} PIN_AB23
    {LEDR[2]} PIN_AC23
    {LEDR[3]} PIN_AD24
    {LEDR[4]} PIN_AG25
    {LEDR[5]} PIN_AF25
    {LEDR[6]} PIN_AE24
    {LEDR[7]} PIN_AF24
    {LEDR[8]} PIN_AB22
    {LEDR[9]} PIN_AC22
    {HEX0[0]} PIN_W17
    {HEX0[1]} PIN_V18
    {HEX0[2]} PIN_AG17
    {HEX0[3]} PIN_AG16
    {HEX0[4]} PIN_AH17
    {HEX0[5]} PIN_AG18
    {HEX0[6]} PIN_AH18
    {HEX1[0]} PIN_AF16
    {HEX1[1]} PIN_V16
    {HEX1[2]} PIN_AE16
    {HEX1[3]} PIN_AD17
    {HEX1[4]} PIN_AE18
    {HEX1[5]} PIN_AE17
    {HEX1[6]} PIN_V17
    {HEX2[0]} PIN_AA21
    {HEX2[1]} PIN_AB17
    {HEX2[2]} PIN_AA18
    {HEX2[3]} PIN_Y17
    {HEX2[4]} PIN_Y18
    {HEX2[5]} PIN_AF18
    {HEX2[6]} PIN_W16
    {HEX3[0]} PIN_Y19
    {HEX3[1]} PIN_W19
    {HEX3[2]} PIN_AD19
    {HEX3[3]} PIN_AA20
    {HEX3[4]} PIN_AC20
    {HEX3[5]} PIN_AA19
    {HEX3[6]} PIN_AD20
    {HEX4[0]} PIN_AD21
    {HEX4[1]} PIN_AG22
    {HEX4[2]} PIN_AE22
    {HEX4[3]} PIN_AE23
    {HEX4[4]} PIN_AG23
    {HEX4[5]} PIN_AF23
    {HEX4[6]} PIN_AH22
    {HEX5[0]} PIN_AF21
    {HEX5[1]} PIN_AG21
    {HEX5[2]} PIN_AF20
    {HEX5[3]} PIN_AG20
    {HEX5[4]} PIN_AE19
    {HEX5[5]} PIN_AF19
    {HEX5[6]} PIN_AB21
}

foreach {sig pin} $pins {
    set_location_assignment $pin -to $sig
    set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to $sig
}

puts "de10_setup.tcl: assignments added. Now run Processing > Start Compilation."
