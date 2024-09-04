include("joJRM.jl")
include("joJIM.jl")
include("joTimeLapse.jl")
include("joTV.jl")

# # hack for now
# import Base.transpose, Base.conj
# transpose(A::joMECurvelet2D) = adjoint(A)
# conj(A::joMECurvelet2D) = A

function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2

    return interval_center .+ randomshift
end

function GenSrcIdxLSRTM(nsrc::Int, batchsize::Int, iter::Int)

    src_list = collect(1:nsrc)

    inds = [zeros(Int, batchsize) for i = 1:iter]
    # random batch of sources
    for i=1:iter
        length(src_list) < batchsize && (src_list = collect(1:nsrc))
        src_list = src_list[randperm(length(src_list))]
        inds[i] = [pop!(src_list) for b=1:batchsize]
    end

    return inds

end

function Patchy(sw::AbstractMatrix{T1}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T}; bulk_min = 36.6f9, bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9, ρw = 501.9f0, ρo = 1053.0f0) where {T1, T}

    ### works for channel problem
    vs = vp./sqrt(3f0)
    sw = T.(sw)
    bulk_sat1 = rho .* (vp.^2f0 - 4f0/3f0 .* vs.^2f0)
    shear_sat1 = rho .* (vs.^2f0)

    patch_temp = bulk_sat1 ./(bulk_min .- bulk_sat1) - 
    bulk_fl1 ./ phi ./ (bulk_min .- bulk_fl1) + 
    bulk_fl2 ./ phi ./ (bulk_min .- bulk_fl2)

    bulk_sat2 = bulk_min./(1f0./patch_temp .+ 1f0)

    bulk_new = 1f0./( (1f0.-sw)./(bulk_sat1+4f0/3f0*shear_sat1) 
    + sw./(bulk_sat2+4f0/3f0*shear_sat1) ) - 4f0/3f0*shear_sat1

    rho_new = rho + phi .* sw * (ρw - ρo)

    Vp_new = sqrt.((bulk_new+4f0/3f0*shear_sat1)./rho_new)
    return Vp_new, rho_new

end

function Patchy(sw::AbstractArray{T1, 3}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T}; bulk_min = 36.6f9, bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9, ρw = 501.9f0, ρo = 1053.0f0) where {T1, T}

    stack = [Patchy(sw[i,:,:], vp, rho, phi; bulk_min=36.6f9, bulk_fl1=bulk_fl1, bulk_fl2=bulk_fl2, ρw = ρw, ρo=ρo) for i = 1:size(sw,1)]
    return [stack[i][1] for i = 1:size(sw,1)], [stack[i][2] for i = 1:size(sw,1)]
end

function Patchy(sw::AbstractMatrix{T1}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T}, d::Tuple{T, T}; bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9,ρw = 7.766f2, ρo = 1.053f3) where {T1, T}

    ### works for Compass 2D model
    n = size(sw)
    capgrid = Int(round(50f0/d[2]))
    vp = vp * 1f3
    vs = vp ./ sqrt(3f0)
    idx_wb = maximum(find_water_bottom(vp.-minimum(vp)))
    idx_ucfmt = find_water_bottom((vp.-3500f0).*(vp.>3500f0))

    bulk_sat1 = rho .* (vp.^2f0 - 4f0/3f0 .* vs.^2f0) * 1f3
    shear_sat1 = rho .* (vs.^2f0) * 1f3

    bulk_min = zeros(Float32,size(bulk_sat1))

    bulk_min[findall(vp.>=3500f0)] .= 5f10   # mineral bulk moduli
    bulk_min[findall(vp.<3500f0)] .= 1.2f0 * bulk_sat1[findall(vp.<3500f0)] # mineral bulk moduli

    patch_temp = bulk_sat1 ./(bulk_min .- bulk_sat1) - bulk_fl1 ./ phi ./ (bulk_min .- bulk_fl1) + bulk_fl2 ./ phi ./ (bulk_min .- bulk_fl2)

    for i = 1:n[1]
        patch_temp[i,idx_ucfmt[i]-capgrid:idx_ucfmt[i]-1] = patch_temp[i,idx_ucfmt[i]:idx_ucfmt[i]+capgrid-1]
    end

    bulk_sat2 = bulk_min./(1f0./patch_temp .+ 1f0)
    bulk_sat2[findall(bulk_sat2-bulk_sat1.>0)] = bulk_sat1[findall(bulk_sat2-bulk_sat1.>0)]

    bulk_new = (bulk_sat1+4f0/3f0*shear_sat1).*(bulk_sat2+4f0/3f0*shear_sat1) ./( (1f0.-T.(sw)).*(bulk_sat2+4f0/3f0*shear_sat1) 
    + T.(sw).*(bulk_sat1+4f0/3f0*shear_sat1) ) - 4f0/3f0*shear_sat1

	bulk_new[:,1:idx_wb] = bulk_sat1[:,1:idx_wb]
    bulk_new[findall(sw.==0)] = bulk_sat1[findall(sw.==0)]
    rho_new = rho + phi .* T.(sw) * (ρw - ρo) / 1f3
    rho_new[findall(sw.==0)] = rho[findall(sw.==0)]
    Vp_new = sqrt.((bulk_new+4f0/3f0*shear_sat1)./rho_new/1f3)
    Vp_new[findall(sw.==0)] = vp[findall(sw.==0)]

    return Vp_new/1f3, rho_new
end
function Patchy(sw::AbstractArray{T1, 3}, vp::AbstractMatrix{T}, rho::AbstractMatrix{T}, phi::AbstractMatrix{T}, d::Tuple{T, T}; bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9,ρw = 7.766f2, ρo = 1.053f3) where {T1, T}

    stack = [Patchy(sw[i,:,:], vp, rho, phi, d; bulk_fl1=bulk_fl1, bulk_fl2=bulk_fl2, ρw = ρw, ρo=ρo) for i = 1:size(sw,1)]
    return [stack[i][1] for i = 1:size(sw,1)], [stack[i][2] for i = 1:size(sw,1)]
end