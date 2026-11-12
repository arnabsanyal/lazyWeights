# ZCU102 Rev 1.0: 300 MHz user clock (Si570 USER_SI570_SYSCLK, power-up default 300 MHz, HP bank 64); everything else
# is on the PS side. (CLK_125 on G21/F21 is in HD bank 47, which cannot drive an MMCM.)
set_property PACKAGE_PIN AL8 [get_ports SYSCLK_P]
set_property PACKAGE_PIN AL7 [get_ports SYSCLK_N]
set_property IOSTANDARD DIFF_SSTL12 [get_ports {SYSCLK_P SYSCLK_N}]
create_clock -period 3.333 -name sysclk [get_ports SYSCLK_P]
