
"""
    logbin(x,P)

calculate a log-binned distribution from an evenly space distribution `P(x)`
where bins that start with `bin_width_start` (keyword argument) increase with
`inrement_factor` (keyword argument)

# Remarks
* `P` has to be a probability density
* `x` has to be equi-distant
"""
function logbin(x, P; increment_factor=sqrt(2), bin_width_start=1)
    @assert length(x)==length(P)

    dx = x[2]-x[1]

    index     = 1
    x_bin = zeros(1)
    P_bin = zeros(1)
    w_bin = ones(1)*floor(Int, bin_width_start)

    bin_counter = 0
    bin_width::Float64 = w_bin[end]
    for (i,p) in enumerate(P)
        # bin_counter counts the number of bins that are merged currently; if
        # this coincides with the target size, then create new bin and reset
        # counter.
        if bin_counter == w_bin[end]
            index += 1
            bin_counter = 0
            bin_width *= increment_factor
            push!(x_bin, 0)
            push!(P_bin, 0)
            push!(w_bin, floor(Int, bin_width))
        end
        bin_counter += 1

        # estimate expectation value of x in log bin
        x_bin[index] += x[i]*P[i]
        # estimate probability weight of bin (sum of equally spaced
        # probabilities)
        P_bin[index] += P[i]
    end

    # normalize x bins (end of expectation value)
    x_bin ./= P_bin

    # transform P into proper probability DENSITY such that <x>=\sum xP(x)dx/sum P(x)dx
    P_bin ./= w_bin

    return x_bin, P_bin, w_bin.*dx
end
logbin(dist; increment_factor=sqrt(2), bin_width_start=1) = logbin(collect(dist.edges[1])[1:end-1], dist.weights, increment_factor=increment_factor, bin_width_start=bin_width_start)


function pbc_index(x,y,Lx,Ly)
    _index = (y-1)*Lx + x
    #implement periodic boundary
    while x<1
        _index += Lx
        x += Lx
    end
    while x>Lx
        _index -= Lx
        x -= Lx
    end
    while y<1
        _index += Lx*Ly
        y += Ly
    end
    while y>Ly
        _index -= Lx*Ly
        y -= Ly
    end
    return _index
end

# correlation function for d-dimensional state but only along x-axis
function correlation_x!(C,state,lags)
    Ls = size(state)
    C .= 0
    mean_state = mean(state)
    L_x   = CartesianIndex(Ls[1],zeros(Int,length(Ls)-1)...)
    for (l,lag) in enumerate(lags)
        lag_x = CartesianIndex(lag,zeros(Int,length(Ls)-1)...)
        for r in CartesianIndices(Ls)
            r_ = r + lag_x
            if r_[1] > Ls[1]
                r_ -= L_x
            end
            C[l] += (state[r]-mean_state)*(state[r_]-mean_state)
        end
    end
    C./=length(state)
end
function correlation_x(state,lags)
    C = zeros(Float64, length(lags))
    correlation_x!(C, state,lags)
    return C
end

