using LightGraphs
include("../src/utils.jl")

"""
    lattice_nearest_neighbor(Lx::Int,Ly::Int)

lattice with nearest neighbor interactions, where nearest neighbor here
also includes diagonal neighbors.
"""
function lattice_nearest_neighbor(Lx::Int,Ly::Int)
    N=Lx*Ly
    lattice=SimpleGraph(N)
    for y in 1:Ly
        for x in 1:Lx
            add_edge!(lattice, pbc_index(x,y,Lx,Ly), pbc_index(x-1,y+1,Lx,Ly))
            add_edge!(lattice, pbc_index(x,y,Lx,Ly), pbc_index(x  ,y+1,Lx,Ly))
            add_edge!(lattice, pbc_index(x,y,Lx,Ly), pbc_index(x+1,y+1,Lx,Ly))
            add_edge!(lattice, pbc_index(x,y,Lx,Ly), pbc_index(x-1,y  ,Lx,Ly))

            add_edge!(lattice, pbc_index(x,y,Lx,Ly), pbc_index(x+1,y  ,Lx,Ly))
            add_edge!(lattice, pbc_index(x,y,Lx,Ly), pbc_index(x-1,y-1,Lx,Ly))
            add_edge!(lattice, pbc_index(x,y,Lx,Ly), pbc_index(x  ,y-1,Lx,Ly))
            add_edge!(lattice, pbc_index(x,y,Lx,Ly), pbc_index(x+1,y-1,Lx,Ly))
        end
    end
    return lattice
end

"""
    step!(rng, state, state_new, lattice, p_rec[, state_rev, val_ref])

recurrent update step with stochastic activations of `state_new` from `state`
based on interactions defined in `lattice` with probability `p_rec` using
random number generator `rng`.

External activations can be added to `state_new` before calling `step!`.

Optionally, one can pass the refractory state `state_rev` that is reinitalized
upon activation to `val_ref`, decreases one value per time step, and makes
neurons only excitable if `state_ref[i]>0`.
"""
function step!(
        rng::AbstractRNG,
        state::Array{Bool},
        state_new::Array{Bool},
        lattice::SimpleGraph{Int64},
        p_rec
    )
    prepare!(rng, state, state_new, lattice, p_rec)

    # swap
    state .= state_new
    state_new .= false
    return nothing
end
function step!(
        rng::AbstractRNG,
        state::Array{Bool},
        state_new::Array{Bool},
        lattice::SimpleGraph{Int64},
        p_rec,
        state_ref::Array{Int},
        val_ref::Int,
    )
    prepare!(rng, state, state_new, lattice, p_rec)

    # if still in refractory period then no update
    for i in 1:length(state)
        if state_ref[i] > 0
            state_new[i] = false
        end
    end

    # swap
    state .= state_new
    state_new .= false

    # update refractory period
    for i in 1:length(state)
        if state[i]
            state_ref[i] = val_ref
        else
            state_ref[i] -= 1
        end
    end
    return nothing
end
function prepare!(rng, state,state_new, lattice, p_rec)
    for i in 1:length(state)
        if state[i]==true
            for j in outneighbors(lattice, i)
                if rand(rng) < p_rec
                    state_new[j] = true
                end
            end
        end
    end
end
