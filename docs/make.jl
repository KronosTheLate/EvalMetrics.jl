using Documenter
using EvalMetrics


makedocs(
    sitename = "EvalMetrics",
    format = Documenter.HTML(prettyurls = false),
    modules = [EvalMetrics],
    pages = [
        "Home" => "index.md",
        "Classification metrics" => "metrics.md",
        "Evaluation curves" => "curves.md",
        "Decision thresholds" => "thresholds.md",
        "Label encodings" => "encodings.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
