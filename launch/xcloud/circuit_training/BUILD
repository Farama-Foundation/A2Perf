load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")

pytype_strict_binary(
    name = "entrypoint",
    srcs = ["entrypoint.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
    ],
)
