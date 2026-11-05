############### Add include ###################
list(APPEND ADD_INCLUDE "include")
list(APPEND ADD_PRIVATE_INCLUDE "")
###############################################

############ Add source files #################
# src/core is plain C (drivers, sampler, UI); src/main_maixcdk.cpp is the SDK shim
append_srcs_dir(ADD_SRCS "src" "src/core")
###############################################

###### Add required/dependent components ######
list(APPEND ADD_REQUIREMENTS basic vision nn peripheral pthread m)
###############################################

register_component()
