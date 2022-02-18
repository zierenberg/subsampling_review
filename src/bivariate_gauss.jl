"""
    bivariate_gauss(rho, num_elements)

returns a bivariate gauss series of lengths `num_elements` with correlation
coefficient `rho`

# Parameters
    * len : seriese length
    * rho : correlation coefficient, 0 <= rho < 1, and rho = exp(-1/tau_exp)
"""
function bivariate_gauss(rng, rho, num_elements::Int)
    series = Vector{Float64}(undef, num_elements)

    series[1] = bivariate_gauss(rng)
    for i in 2:num_elements
        series[i] = bivariate_gauss(rng, rho, series[i-1])
    end

    return series
end

function bivariate_gauss(rng, rho, x_::Float64)
    return rho*x_ + sqrt(1-rho^2)*randn(rng)
end
function bivariate_gauss(rng)
    return randn(rng)
end
