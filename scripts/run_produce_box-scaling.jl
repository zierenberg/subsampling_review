include("./produce_data.jl")
using NPZ

# generate all data sets for paper
function main()
    h=1e-7
    ms=[1.09, 1.10, 1.20]
    for m in ms
        println("m=",m)
        data=produce_data_box_scaling_branching_network(m, h)
        npzwrite(@sprintf("%s/data/data_box_scaling_m%06.4f_h%.2e.npz", path_project,m,h),data)
    end
end

# call function
main()


