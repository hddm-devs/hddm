import pycuda.driver as cuda
import pycuda.compiler
import pycuda.autoinit

import numpy as np
import numpy.testing

x = np.random.rand(100).astype(np.float32)

#ctx = cl.create_some_context()
#queue = cl.CommandQueue(ctx)

#mf = cl.mem_flags
#x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
#dest = np.float()
#dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, dest.__sizeof__())

#prg = cl.Program(ctx, """
kernel_source = """
    __global__ void pdf(const float *x, float const a, float const z_val, float const v_val, float const err, int const logp, float *out)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        float z, v, t;
        
        if (x[idx] > 0) {
            z = a - z_val;
            v = -v_val;
            t = x[idx];
        }
        else {
            z = z_val;
            v = v_val;
            t = -x[idx];
        }

        float tt = t/(powf(a,2)); // use normalized time
        float w = z/a; // convert to relative start point
        float kl, ks, p;
        float PI = 3.1415926535897f;
        float PIs = 9.869604401089358f; // PI^2
        int k, K, lower, upper;


        // calculate number of terms needed for large t
        if (PI*tt*err<1) { // if error threshold is set low enough
            kl=sqrtf(-2*logf(PI*tt*err)/(PIs*tt)); // bound
            kl=fmax(kl,1/(PI*sqrtf(tt))); // ensure boundary conditions met
        }
        else { // if error threshold set too high
            kl=1/(PI*sqrtf(tt)); // set to boundary condition
        }


        // calculate number of terms needed for small t
        if (2*sqrtf(2*PI*tt)*err<1) { // if error threshold is set low enough
            ks=2+sqrtf(-2*tt*logf(2*sqrtf(2*PI*tt)*err)); // bound
            ks=fmax(ks,sqrtf(tt)+1); // ensure boundary conditions are met
        }
        else { // if error threshold was set too high
            ks=2; // minimal kappa for that case
        }

        // compute f(tt|0,1,w)
        p=0; //initialize density
        if (ks<kl) { // if small t is better (i.e., lambda<0)
            K=(int)(ceilf(ks)); // round to smallest integer meeting error
            lower = (int)(-floorf((K-1)/2.));
            upper = (int)(ceilf((K-1)/2.));
        
            for (k=lower; k <= upper; k++) { // loop over k
                p=p+(w+2*k)*expf(-(powf((w+2*k),2))/2/tt); // increment sum
            }
            p=p/sqrtf(2*PI*powf(tt,3)); // add constant term
        }
        else { // if large t is better...
            K=(int)(ceilf(kl)); // round to smallest integer meeting error
            for(k=1; k <= K; k++) {
                p=p+k*expf(-(powf(k,2))*(PIs)*tt/2)*sinf(k*PI*w); // increment sum
            }
            p=p*PI; // add constant term
        }
        // convert to f(t|v,a,w)
        if (logp == 0) {
            out[idx] = p*expf(-v*a*w -(powf(v,2))*t/2.)/(powf(a,2));
        }
        else { 
            out[idx] = logf(p) + (-v*a*w -(powf(v,2))*t/2.) - 2*logf(a);
        }
    }
    """#).build()

kernel_source_complete = """
    __global__ void pdf(const float *x, float const *a_buf, float const *z_buf, float const *v_buf, float const err, int const logp, float *out)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        float tu = x[idx];
        float a, z, v, t;
        // if t is negative, lower boundary, and vice versa
        if (tu < 0) {
            a = a_buf[idx];
            z = z_buf[idx];
            v = v_buf[idx];
            t = -tu;
        }
        else {
            a = a_buf[idx];
            z = a_buf[idx] - z_buf[idx];
            v = -v_buf[idx];
            t = tu;
        }

        float tt = t/(powf(a,2)); // use normalized time
        float w = z/a; // convert to relative start point
        float kl, ks, p;
        float PI = 3.1415926535897f;
        float PIs = 9.869604401089358f; // PI^2
        int k, K, lower, upper;

        // calculate number of terms needed for large t
        if (PI*tt*err<1) { // if error threshold is set low enough
            kl=sqrtf(-2*logf(PI*tt*err)/(PIs*tt)); // bound
            kl=fmax(kl,1/(PI*sqrtf(tt))); // ensure boundary conditions met
        }
        else { // if error threshold set too high
            kl=1/(PI*sqrtf(tt)); // set to boundary condition
        }


        // calculate number of terms needed for small t
        if (2*sqrtf(2*PI*tt)*err<1) { // if error threshold is set low enough
            ks=2+sqrtf(-2*tt*logf(2*sqrtf(2*PI*tt)*err)); // bound
            ks=fmax(ks,sqrtf(tt)+1); // ensure boundary conditions are met
        }
        else { // if error threshold was set too high
            ks=2; // minimal kappa for that case
        }

        // compute f(tt|0,1,w)
        p=0; //initialize density
        if (ks<kl) { // if small t is better (i.e., lambda<0)
            K=(int)(ceilf(ks)); // round to smallest integer meeting error
            lower = (int)(-floorf((K-1)/2.));
            upper = (int)(ceilf((K-1)/2.));
        
            for (k=lower; k <= upper; k++) { // loop over k
                p=p+(w+2*k)*expf(-(powf((w+2*k),2))/2/tt); // increment sum
            }
            p=p/sqrtf(2*PI*powf(tt,3)); // add constant term
        }
        else { // if large t is better...
            K=(int)(ceilf(kl)); // round to smallest integer meeting error
            for(k=1; k <= K; k++) {
                p=p+k*expf(-(powf(k,2))*(PIs)*tt/2)*sinf(k*PI*w); // increment sum
            }
            p=p*PI; // add constant term
        }
        // convert to f(t|v,a,w)
        if (logp == 0) {
            out[idx] = p*expf(-v*a*w -(powf(v,2))*t/2.)/(powf(a,2));
        }
        else { 
            out[idx] = logf(p) + (-v*a*w -(powf(v,2))*t/2.) - 2*logf(a);
        }
    }
    """#).build()

pdf = pycuda.compiler.SourceModule(kernel_source)
pdf_func = pdf.get_function("pdf")

pdf_complete=pycuda.compiler.SourceModule(kernel_source_complete)
pdf_func_complete = pdf_complete.get_function("pdf")


dest_buf = np.empty_like(x)

pdf_func(cuda.In(x), np.float32(2.), np.float32(1.), np.float32(.5), np.float32(0.0001), np.int16(1), cuda.Out(dest_buf), block=(x.shape[0], 1, 1))

#print dest_buf

dest_buf_complete = np.empty_like(x)

pdf_func_complete(cuda.In(-x), cuda.In(np.ones_like(x)*2), cuda.In(np.ones_like(x)*1.), cuda.In(np.ones_like(x)*.5), np.float32(0.0001), np.int16(1), cuda.Out(dest_buf_complete), block=(x.shape[0], 1, 1))

#print dest_buf_complete

#print "Testing for equality"
#np.testing.assert_array_almost_equal(dest_buf, dest_buf_complete)

#x_out = np.empty_like(x)
#cl.enqueue_read_buffer(queue, dest_buf, x_out).wait()

kernel_source_complete_opt = """
    __global__ void pdf(const float *x, float const *a_buf, float const *z_buf, float const *v_buf, float const err, int const logp, int bias_z, float *out)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        float tu = x[idx];
        float a, z, v, t;

        a = a_buf[idx];
        if (bias_z == 1) {
            z = a/2;
        }
        else {
            z = z_buf[idx];
        }
        v = v_buf[idx];
        t = tu;

        // if t is negative, lower boundary, and vice versa
        //if (t == -9999 || a<z || a<0 || z<0) {
        //    out[idx] = -9999;
        //    return;
        //}
        
        float tt = t/(powf(a,2)); // use normalized time
        float w = z/a; // convert to relative start point
        float kl, ks, p;
        float PI = 3.1415926535897f;
        float PIs = 9.869604401089358f; // PI^2
        int k, K, lower, upper;

        // calculate number of terms needed for large t
        if (PI*tt*err<1) { // if error threshold is set low enough
            kl=sqrtf(-2*logf(PI*tt*err)/(PIs*tt)); // bound
            kl=fmax(kl,1/(PI*sqrtf(tt))); // ensure boundary conditions met
        }
        else { // if error threshold set too high
            kl=1/(PI*sqrtf(tt)); // set to boundary condition
        }


        // calculate number of terms needed for small t
        if (2*sqrtf(2*PI*tt)*err<1) { // if error threshold is set low enough
            ks=2+sqrtf(-2*tt*logf(2*sqrtf(2*PI*tt)*err)); // bound
            ks=fmax(ks,sqrtf(tt)+1); // ensure boundary conditions are met
        }
        else { // if error threshold was set too high
            ks=2; // minimal kappa for that case
        }

        // compute f(tt|0,1,w)
        p=0; //initialize density
        if (ks<kl) { // if small t is better (i.e., lambda<0)
            K=(int)(ceilf(ks)); // round to smallest integer meeting error
            lower = (int)(-floorf((K-1)/2.));
            upper = (int)(ceilf((K-1)/2.));
        
            for (k=lower; k <= upper; k++) { // loop over k
                p=p+(w+2*k)*expf(-(powf((w+2*k),2))/2/tt); // increment sum
            }
            p=p/sqrtf(2*PI*powf(tt,3)); // add constant term
        }
        else { // if large t is better...
            K=(int)(ceilf(kl)); // round to smallest integer meeting error
            for(k=1; k <= K; k++) {
                p=p+k*expf(-(powf(k,2))*(PIs)*tt/2)*sinf(k*PI*w); // increment sum
            }
            p=p*PI; // add constant term
        }
        // convert to f(t|v,a,w)
        if (logp == 0) {
            out[idx] = p*expf(-v*a*w -(powf(v,2))*t/2.)/(powf(a,2));
        }
        else { 
            out[idx] = logf(p) + (-v*a*w -(powf(v,2))*t/2.) - 2*logf(a);
        }
    }
    """#).build()

pdf_opt=pycuda.compiler.SourceModule(kernel_source_complete_opt)
pdf_func_opt = pdf_opt.get_function("pdf")

kernel_source_complete2 = """
    __global__ void pdf(const float *x, float const *a_buf, float const *z_buf, float const *v_buf, float const err, int const logp, float *out)
    {
        int idx = threadIdx.x;
        float tu = x[idx];
        float a, z, v, t;
        // if t is negative, lower boundary, and vice versa
        if (tu < 0) {
            a = a_buf[idx];
            z = z_buf[idx];
            v = v_buf[idx];
            t = abs(tu);
        }
        else {
            a = a_buf[idx];
            z = a_buf[idx] - z_buf[idx];
            v = -v_buf[idx];
            t = tu;
        }

        if (t < 0) {
            out[idx] = -9999;
            return;
        }
        
        float tt = t/(powf(a,2)); // use normalized time
        float w = z/a; // convert to relative start point
        float kl, ks, p;
        float PI = 3.1415926535897f;
        float PIs = 9.869604401089358f; // PI^2
        int k, K, lower, upper;

        // calculate number of terms needed for large t
        if (PI*tt*err<1) { // if error threshold is set low enough
            kl=sqrtf(-2*logf(PI*tt*err)/(PIs*tt)); // bound
            kl=fmax(kl,1/(PI*sqrtf(tt))); // ensure boundary conditions met
        }
        else { // if error threshold set too high
            kl=1/(PI*sqrtf(tt)); // set to boundary condition
        }


        // calculate number of terms needed for small t
        if (2*sqrtf(2*PI*tt)*err<1) { // if error threshold is set low enough
            ks=2+sqrtf(-2*tt*logf(2*sqrtf(2*PI*tt)*err)); // bound
            ks=fmax(ks,sqrtf(tt)+1); // ensure boundary conditions are met
        }
        else { // if error threshold was set too high
            ks=2; // minimal kappa for that case
        }

        // compute f(tt|0,1,w)
        p=0; //initialize density
        if (ks<kl) { // if small t is better (i.e., lambda<0)
            K=(int)(ceilf(ks)); // round to smallest integer meeting error
            lower = (int)(-floorf((K-1)/2.));
            upper = (int)(ceilf((K-1)/2.));
        
            for (k=lower; k <= upper; k++) { // loop over k
                p=p+(w+2*k)*expf(-(powf((w+2*k),2))/2/tt); // increment sum
            }
            p=p/sqrtf(2*PI*powf(tt,3)); // add constant term
        }
        else { // if large t is better...
            K=(int)(ceilf(kl)); // round to smallest integer meeting error
            for(k=1; k <= K; k++) {
                p=p+k*expf(-(powf(k,2))*(PIs)*tt/2)*sinf(k*PI*w); // increment sum
            }
            p=p*PI; // add constant term
        }
        // convert to f(t|v,a,w)
        if (logp == 0) {
            out[idx] = p*expf(-v*a*w -(powf(v,2))*t/2.)/(powf(a,2));
        }
        else { 
            out[idx] = logf(p) + (-v*a*w -(powf(v,2))*t/2.) - 2*logf(a);
        }
    }
    """#).build()

kernel_source_complete_ter = """
    __global__ void pdf(const float *x, float const *a_buf, float const *z_buf, float const *v_buf, float const *ter_buf, float const err, int const logp, float *out)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        float tu = x[idx];
        float a, z, v, t, ter;
        // if t is negative, lower boundary, and vice versa
        ter = ter_buf[idx];
        a = a_buf[idx];
        if (tu < 0) {
            z = z_buf[idx];
            v = v_buf[idx];
            t = -tu;
        }
        else {
            z = a_buf[idx] - z_buf[idx];
            v = -v_buf[idx];
            t = tu;
        }
        //if (ter>t) {return -9999}
        t = t-ter;
        
        float tt = t/(powf(a,2)); // use normalized time
        float w = z/a; // convert to relative start point
        float kl, ks, p;
        float PI = 3.1415926535897f;
        float PIs = 9.869604401089358f; // PI^2
        int k, K, lower, upper;

        // calculate number of terms needed for large t
        if (PI*tt*err<1) { // if error threshold is set low enough
            kl=sqrtf(-2*logf(PI*tt*err)/(PIs*tt)); // bound
            kl=fmax(kl,1/(PI*sqrtf(tt))); // ensure boundary conditions met
        }
        else { // if error threshold set too high
            kl=1/(PI*sqrtf(tt)); // set to boundary condition
        }


        // calculate number of terms needed for small t
        if (2*sqrtf(2*PI*tt)*err<1) { // if error threshold is set low enough
            ks=2+sqrtf(-2*tt*logf(2*sqrtf(2*PI*tt)*err)); // bound
            ks=fmax(ks,sqrtf(tt)+1); // ensure boundary conditions are met
        }
        else { // if error threshold was set too high
            ks=2; // minimal kappa for that case
        }

        // compute f(tt|0,1,w)
        p=0; //initialize density
        if (ks<kl) { // if small t is better (i.e., lambda<0)
            K=(int)(ceilf(ks)); // round to smallest integer meeting error
            lower = (int)(-floorf((K-1)/2.));
            upper = (int)(ceilf((K-1)/2.));
        
            for (k=lower; k <= upper; k++) { // loop over k
                p=p+(w+2*k)*expf(-(powf((w+2*k),2))/2/tt); // increment sum
            }
            p=p/sqrtf(2*PI*powf(tt,3)); // add constant term
        }
        else { // if large t is better...
            K=(int)(ceilf(kl)); // round to smallest integer meeting error
            for(k=1; k <= K; k++) {
                p=p+k*expf(-(powf(k,2))*(PIs)*tt/2)*sinf(k*PI*w); // increment sum
            }
            p=p*PI; // add constant term
        }
        // convert to f(t|v,a,w)
        if (logp == 0) {
            out[idx] = p*expf(-v*a*w -(powf(v,2))*t/2.)/(powf(a,2));
        }
        else { 
            out[idx] = logf(p) + (-v*a*w -(powf(v,2))*t/2.) - 2*logf(a);
        }
    }
    """#).build()

pdf_ter=pycuda.compiler.SourceModule(kernel_source_complete_ter)
pdf_func_ter = pdf_ter.get_function("pdf")
