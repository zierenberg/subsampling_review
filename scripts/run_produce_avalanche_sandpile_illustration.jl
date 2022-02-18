include("./produce_data.jl")
using NPZ
using HDF5

function main(;
    )
    # create data
    data = produce_data_avalanches_sandpile_illustration()
    npzwrite(@sprintf("%s/data/data_avalanche_sandpile_illustration.npz", path_project),data)
end

# call function
main()


