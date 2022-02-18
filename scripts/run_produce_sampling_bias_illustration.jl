include("./produce_data.jl")
using NPZ
using HDF5

function main(;
    )
    # create data
    data = produce_data_sampling_bias_illustration()
    npzwrite(@sprintf("%s/data/data_sampling_bias_illustration.npz", path_project),data)
end

# call function
main()


