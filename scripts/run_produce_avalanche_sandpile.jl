include("./produce_data.jl")
using NPZ
using HDF5

function dist_logbin(sizes, start_index, increment_log_bin)
    dist_size = normalize!(float(fit(Histogram, sizes[start_index:end], 1:maximum(sizes))))
    return logbin(dist_size.edges[1][1:end-1], dist_size.weights, increment_factor=increment_log_bin)
end

function main(;
        check_equilibration::Bool = true,
        minimum_number_avalanches::Int = Int(1e4),
        rolling_window::Int = Int(1e3),
        increment_log_bin::Float64 = sqrt(1.1)
    )
    # create data
    subsample, data = produce_data_avalanches_sandpile()

    if check_equilibration
        println("# determine initial equilibration period (will be discarded) from mean size of avalanches")
        mean_size = rollmean(data["size"]["full"], rolling_window)
        start_index = floor(Int, 1.5 * findfirst(mean_size .> mean(mean_size[floor(Int, length(mean_size)/2):end])))
    else
        start_index = 1
    end

    if length(data["size"]["full"]) - start_index > minimum_number_avalanches
        println("# enough data ... create output file")
        filename_out = @sprintf("%s/data/data_avalanche_sandpile.h5", path_project)
        fid = h5open(filename_out, "w")
        close(fid)
        println("# full sample")
        sizes = data["size"]["full"]
        x,P = dist_logbin(sizes, start_index, increment_log_bin)

        datasetname = "size/full"
        h5write(filename_out, datasetname, hcat(x,P))
        h5writeattr(filename_out, datasetname, Dict("description"=>"avalanche size distribution of full system (x,P)"))

        durations = data["duration"]["full"]
        x,P = dist_logbin(durations, start_index, increment_log_bin)
        datasetname = "duration/full"
        h5write(filename_out, datasetname, hcat(x,P))
        h5writeattr(filename_out, datasetname, Dict("description"=>"avalanche duration distribution of full system (x,P)"))

        # subsamples
        println("# subsamples")
        for (i,l) in enumerate(subsample["l"])
            x,P = dist_logbin(data["size"]["random"][i], start_index, increment_log_bin)
            datasetname = @sprintf("size/random/%d",l)
            h5write(filename_out, datasetname, hcat(x,P))
            h5writeattr(filename_out, datasetname, Dict("description"=>@sprintf("avalanche size distribution of randomly subsampled (n=%dx%d) systems (x,P)", l,l)))

            x,P = dist_logbin(data["size"]["window"][i], start_index, increment_log_bin)
            datasetname = @sprintf("size/window/%d",l)
            h5write(filename_out, datasetname, hcat(x,P))
            h5writeattr(filename_out, datasetname, Dict("description"=>@sprintf("avalanche size distribution of finite-field-of-view subsampled (%dx%d in center) systems (x,P)", l,l)))
        end
    end
end

# call function
main()


