using Pkg
Pkg.add(["XLSX", "DataFrames", "Plots", "StatsBase"])

using XLSX
using DataFrames
using Plots
using StatsBase

try
    global df = DataFrame(XLSX.readtable("../data/cableTV.xlsx", "cableTV", infer_eltypes=true))
    println("File read successfully")
    global read_success = true
catch e
    println("Error while reading the file: ", e)
    global read_success = false
end

if read_success
    println("--------------------------------")
    println("Dataframe head:")
    println(first(df, 10))
end

if read_success
    println("--------------------------------")
    println("Dataframe info:")
    println("Shape: ", size(df))
    println("Column names: ", names(df))
    println("Column types: ", eltype.(eachcol(df)))
end

if read_success
    println("--------------------------------")
    println("Dataframe describe:")
    println(describe(df))
end

if read_success
    println("--------------------------------")
    println("Dataframe shape:")
    println(size(df))
end

if read_success
    println("--------------------------------")
    println("Dataframe missing values:")
    for col in names(df)
        missing_count = sum(ismissing.(df[!, col]))
        println("$col: $missing_count")
    end
    println("--------------------------------")
    println("Dataframe unique values:")
    for col in names(df)
        unique_count = length(unique(skipmissing(df[!, col])))
        println("$col: $unique_count")
    end
end

if read_success
    quant_vars = ["adultos", "ninos", "teles", "tvtot", "renta", "valor"]

    for var in quant_vars
        values = df[!, var]
        label = var

        if var == "valor"
            values = values ./ 1000
            label *= " (thousands of pesos)"
        end

        p = histogram(values,
            bins=:auto,
            xlabel=label,
            ylabel="Frequency",
            title="Frequency for $label",
            grid=true,
            gridwidth=1,
            gridcolor=:gray,
            gridalpha=0.5,
            linecolor=:black,
            linewidth=1)

        savefig(p, "../plots/julia/frequency_$var.png")
    end
end

if read_success
    cat_vars = ["colonia", "tipo"]

    for var in cat_vars
        println("=== Frequency of '$var' ===")
        freq_table = countmap(skipmissing(df[!, var]))
        for (key, value) in sort(collect(freq_table), by=x -> x[2], rev=true)
            println("$key: $value")
        end
        println()
    end
end
