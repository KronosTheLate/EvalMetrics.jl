using Documenter
using EvalMetrics

DocMeta.setdocmeta!(EvalMetrics, :DocTestSetup, :(using EvalMetrics); recursive=true)

makedocs(
    sitename = "EvalMetrics",
    format = Documenter.HTML(prettyurls = false),
    modules = [EvalMetrics],
    pages = [
        "Home" => "index.md",
        "Classification metrics" => "metrics.md",
        "Decision thresholds" => "thresholds.md",
        "Evaluation curves" => "curves.md",
        "Label encodings" => "encodings.md",
        "API" => "api.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
