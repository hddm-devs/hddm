import hddm

def parse_config_file(fname, load=False):
    import ConfigParser
    
    config = ConfigParser.ConfigParser()
    config.read(fname)
    
    #####################################################
    # Parse config file
    data_fname = config.get('data', 'load')
    save = config.get('data', 'save')
    #format_ = config.get('data', 'format')

    data = np.recfromcsv(data_fname)
    
    try:
        model_type = config.get('model', 'type')
    except ConfigParser.NoOptionError:
        model_type = 'simple'

    try:
        is_subj_model = config.getboolean('model', 'is_subj_model')
    except ConfigParser.NoOptionError:
        is_subj_model = True

    try:
        no_bias = config.getboolean('model', 'no_bias')
    except ConfigParser.NoOptionError:
        no_bias = True

    try:
        debug = config.getboolean('model', 'debug')
    except ConfigParser.NoOptionError:
        debug = False

    try:
        dbname = config.get('mcmc', 'dbname')
    except ConfigParser.NoOptionError:
        dbname = None

    if model_type == 'simple' or model_type == 'simple_gpu':
        group_param_names = ['a', 'v', 'z', 't']
    elif model_type == 'full_avg' or model_type == 'full':
        group_param_names = ['a', 'v', 'V', 'z', 'Z', 't', 'T']
    elif model_type == 'lba':
        group_param_names = ['a', 'v', 'z', 't', 'V']
    else:
        raise NotImplementedError('Model type %s not implemented'%model_type)

    # Get depends
    depends = {}
    for param_name in group_param_names:
        try:
            # Multiple depends can be listed (separated by a comma)
            depends[param_name] = config.get('depends', param_name).split(',')
        except ConfigParser.NoOptionError:
            pass

    # MCMC values
    try:
        samples = config.getint('mcmc', 'samples')
    except ConfigParser.NoOptionError:
        samples = 10000
    try:
        burn = config.getint('mcmc', 'burn')
    except ConfigParser.NoOptionError:
        burn = 5000
    try:
        thin = config.getint('mcmc', 'thin')
    except ConfigParser.NoOptionError:
        thin = 3
    try:
        verbose = config.getint('mcmc', 'verbose')
    except ConfigParser.NoOptionError:
        verbose = 0

    print "Creating model..."
    m = Multi(data, model_type=model_type, is_subj_model=is_subj_model, no_bias=no_bias, depends_on=depends, save_stats_to=save, debug=debug)

    if not load:
        print "Sampling... (this can take some time)"
        m.mcmc(samples=samples, burn=burn, thin=thin, verbose=verbose, dbname=dbname)
    else:
        m.mcmc_load_from_db(dbname=dbname)

    return m

if __name__=='__main__':
    import sys
    parse_config_file(sys.argv[1])
