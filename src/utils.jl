function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--dt"
            help = "time-step of jutul simulation"
            arg_type = Float64
            default = 365.25
        "--nt"
            help = "Number of time-step of jutul simulation"
            arg_type = Int
            default = 10
        "--irate"
            help = "Injectivity of CO2 for jutul simulation"
            arg_type = Float64
            default = 8e-2        
        "--timeD"
            help = "length of seismic data"
            arg_type = Float32
            default = 3000f0
        "--dtD"
            help = "time sampling of seismic data"
            arg_type = Float32
            default = 2f0
        "--f0"
            help = "central frequency for seismic simulation"
            arg_type = Float32
            default = 0.0145f0
        "--nv"
            help = "Total number of vintage"
            arg_type = Int
            default = 6
        "--nv_id"
            help = "Index of monitor vintage to be inverted (2 - 11)"
            arg_type = Int
            default = 2
        "--nsrc"
            help = "Number of sources per vintage"
            arg_type = Int
            default = 240
        "--nrec"
            help = "Number of sources per vintage"
            arg_type = Int
            default = 120
        "--niter"
            help = "JRM iterations"
            arg_type = Int
            default = 30
        "--batchsize"
            help = "batchsize in JRM iterations"
            arg_type = Int
            default = 12
        "--snr"
            help = "SNR of noisy data"
            arg_type = Float64
            default = 10.0
        "--gamma"
            help = "Weighting on common component"
            arg_type = Float64
            default = 1.0
        "--inv_type"
            help = "Type of inversion - joint or independent"
            arg_type = String
            default = "joint"
        "--replicate"
            help = "Replicated survey of non-replicated"
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end