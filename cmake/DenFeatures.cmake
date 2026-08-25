# Den compile-time feature gates.
#
# Build profiles:
#   den-full  : all Den features ON (default)
#   den-clean : upstream + DFlash2 only  (-DDEN_KVARN=OFF -DDEN_NVFP4_KV=OFF
#              -DDEN_QWEN35_MTP=OFF -DDEN_ESCHA=OFF)
#   den-min   : baseline, no DFlash2 either (add -DDEN_DFLASH2=OFF; not yet
#               fully supported, see warning below)
#
# Each option adds a matching DEN_* preprocessor definition used by the
# #ifdef guards in the shared sources.

option(DEN_KVARN      "Den kvarn/NVFP4-KV cache"     ON)
option(DEN_NVFP4_KV   "Den NVFP4-KV quantization"    ON)
option(DEN_QWEN35_MTP "Den qwen35 MTP head"          ON)
option(DEN_ESCHA      "Den Escha W2 dense path"      ON)
option(DEN_GDN_FAST   "Den GDN fast exp2"            OFF)

# pr27342 (DFlash2/DSpark), not a Den modification — always ON for now.
option(DEN_DFLASH2    "DFlash2 speculative decoding (pr27342)" ON)

if(DEN_KVARN)
    add_compile_definitions(DEN_KVARN)
endif()

if(DEN_NVFP4_KV)
    add_compile_definitions(DEN_NVFP4_KV)
endif()

if(DEN_QWEN35_MTP)
    add_compile_definitions(DEN_QWEN35_MTP)
endif()

if(DEN_ESCHA)
    add_compile_definitions(DEN_ESCHA)
endif()

# Compile-time strip of the runtime env-var opt-in in gated_delta_net.cu.
# The code checks DEN_GDN_FAST_EXP; keep the macro name aligned with it.
if(DEN_GDN_FAST)
    add_compile_definitions(DEN_GDN_FAST_EXP)
endif()

if(NOT DEN_DFLASH2)
    message(WARNING "DEN_DFLASH2=OFF: DFlash2 code paths are not yet fully "
        "gated; this configuration is not supported.")
endif()
add_compile_definitions(DEN_DFLASH2)
