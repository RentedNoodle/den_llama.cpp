# mtmd needs these even when common is not built
add_subdirectory(hash)
add_subdirectory(miniaudio)
add_subdirectory(nlohmann)
add_subdirectory(sheredom)
add_subdirectory(stb)

# only used by common
if (LLAMA_BUILD_COMMON)
    add_subdirectory(cpp-httplib)
    # n-gram cache fast paths (jadidbourbaki PRs), gated by DEN_NGRAM_FAST
    add_subdirectory(ankerl)
    add_subdirectory(constmap)
endif()
