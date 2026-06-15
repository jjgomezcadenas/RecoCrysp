using Documenter
using RecoCrysp

makedocs(
    sitename = "RecoCrysp.jl",
    modules = [RecoCrysp],
    format = Documenter.HTML(prettyurls = false),
    remotes = nothing,  # local package without a GitHub remote (yet)
    pages = [
        "Home" => "index.md",
        "Backends and GPU usage" => "backends.md",
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
)
