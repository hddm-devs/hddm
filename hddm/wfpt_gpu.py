import hddm

import pycuda.driver as cuda
import pycuda.compiler
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import numpy as np
import numpy.testing

#print "Testing for equality"
#np.testing.assert_array_almost_equal(dest_buf, dest_buf_complete)

#x_out = np.empty_like(x)
#cl.enqueue_read_buffer(queue, dest_buf, x_out).wait()


#pdf_opt=pycuda.compiler.SourceModule(kernel_source_complete_opt)
#pdf_func_opt = pdf_opt.get_function("pdf")
from pycuda.elementwise import ElementwiseKernel

pdf_gpu_tmp = ElementwiseKernel(
    "float *x, float v, float a, float z, float ter, float err, float *out",
    "out[i] = pdf(x[i], v, a, z, ter, err)",
    "wfpt_gpu",
    preamble=
    """
    #include <math_constants.h>

/*    __device__ float ftt_01w(double tt, double w, double err)
    {
    float kl, ks, p;
    int k, K, lower, upper;

    // calculate number of terms needed for large t
    if (CUDART_PI_F*tt*err<1) { // if error threshold is set low enough
        kl=sqrt(-2*log(CUDART_PI_F*tt*err)/(pow(CUDART_PI_F,2)*tt)); // bound
        kl=fmax(kl,1/(CUDART_PI_F*sqrtf(tt))); // ensure boundary conditions met
    }
    else { // if error threshold set too high
        kl=1./(CUDART_PI_F*sqrt(tt)); // set to boundary condition
    }

    // calculate number of terms needed for small t
    if (2*sqrt(2*CUDART_PI_F*tt)*err<1) { // if error threshold is set low enough
        ks=2+sqrt(-2*tt*log(2*sqrt(2*CUDART_PI_F*tt)*err)); // bound
        ks=fmax(ks,sqrtf(tt)+1); // ensure boundary conditions are met
    }
    else { // if error threshold was set too high
        ks=2; // minimal kappa for that case
    }

    // compute f(tt|0,1,w)
    p=0; //initialize density
    if (ks<kl) { // if small t is better (i.e., lambda<0)
        K=(int)(ceil(ks)); // round to smallest integer meeting error
        lower = (int)(-floor((K-1)/2.));
        upper = (int)(ceil((K-1)/2.));
        for(k=lower; k<=upper; k++) {// loop over k
            p+=(w+2*k)*exp(-(pow((w+2*k),2))/2/tt); // increment sum
        }
        p/=sqrt(2*CUDART_PI_F*pow(tt,3)); // add constant term
    }
    else { // if large t is better...
        K=(int)(ceil(kl)); // round to smallest integer meeting error
        for (k=1; k <= K; k++) {
            p+=k*exp(-(pow(k,2))*(pow(CUDART_PI_F,2))*tt/2)*sin(k*CUDART_PI_F*w); // increment sum
        }
        p*=CUDART_PI_F; // add constant term
    }
    return p;
    }*/

    __device__ float pdf(float x, float v_in, float a, float w_in, float ter, float err)
    {
        float w, v, t;
        // if t is negative, lower boundary, and vice versa
        if (x < 0) {
            w = w_in;
            v = v_in;
            t = -x;
        }
        else {
            w = 1-w_in;
            v = -v_in;
            t = x;
        }
        if (ter>t) {return (-9999.0f);}
        
        // Subtact ter
        //t -= ter;

        double tt = x/(powf(a,2)); // use normalized time
        double p = 0.5f;
        //double p = ftt_01w(tt, w, err); //get f(t|0,1,w)
  
        // convert to f(t|v,a,w)
        return p*expf(-v*a*w -(powf(v,2.0f))*x/2.)/(powf(a,2.0f));
    }
    """)

pdf_gpu = ElementwiseKernel(
    "const float *x, const float v, const float a, const float z, const float ter, float err, float *out",
    "out[i] = pdf(x[i], v, a, z, ter, err)",
    "wfpt_gpu",
    preamble=
    """
    #include <math_constants.h>
    __device__ float pdf(const float x, const float v_in, const float a, const float z_in, const float ter, const float err)
    {
        float w, v, t;
        // if t is negative, lower boundary, and vice versa
        if (x < 0) {
            w = z_in;
            v = v_in;
            t = -x;
        }
        else {
            w = 1-z_in;
            v = -v_in;
            t = x;
        }
        if (ter>t) {return (0.0f);}
        
        // Subtract ter
        t -= ter;

        float tt = t/(powf(a,2)); // use normalized time
        //float w = z; //z/a; // convert to relative start point
        float kl, ks, p;
        float PI = CUDART_PI_F; //3.1415926535897f;
        float PIs = powf(PI, 2); //9.869604401089358f; // PI^2
        int k, K, lower, upper;

        // calculate number of terms needed for large t
        if (CUDART_PI_F*tt*err<1) { // if error threshold is set low enough
            kl=sqrtf(-2.0f*log(CUDART_PI_F*tt*err)/(powf(CUDART_PI_F,2)*tt)); // bound
            kl=fmax(kl,1/(CUDART_PI_F*sqrtf(tt))); // ensure boundary conditions met
        }
        else { // if error threshold set too high
            kl=1.0f/(CUDART_PI_F*sqrtf(tt)); // set to boundary condition
        }
    
        // calculate number of terms needed for small t
        if (2*sqrtf(2*CUDART_PI_F*tt)*err<1) { // if error threshold is set low enough
            ks=2+sqrtf(-2*tt*log(2*sqrtf(2*CUDART_PI_F*tt)*err)); // bound
            ks=fmax(ks,sqrtf(tt)+1); // ensure boundary conditions are met
        }
        else { // if error threshold was set too high
            ks=2; // minimal kappa for that case
        }
    
        // compute f(tt|0,1,w)
        p=0.0f; //initialize density
        if (ks<kl) { // if small t is better (i.e., lambda<0)
            K=(int)(ceil(ks)); // round to smallest integer meeting error
            lower = (int)(-floorf((K-1)/2.));
            upper = (int)(ceilf((K-1)/2.));
            for(k=lower; k<=upper; k++) {// loop over k
                p+=(w+2*k)*expf(-(powf((w+2*k),2))/2/tt); // increment sum
            }
            p/=sqrtf(2*CUDART_PI_F*powf(tt,3)); // add constant term
        }
        else { // if large t is better...
            K=(int)(ceil(kl)); // round to smallest integer meeting error
            for (k=1; k <= K; k++) {
                p+=k*expf(-(powf(k,2))*(powf(CUDART_PI_F,2))*tt/2)*sin(k*CUDART_PI_F*w); // increment sum
            }
        p*=CUDART_PI_F; // add constant term
        }

        // convert to f(t|v,a,w)
        p *= expf(-v*a*w -(powf(v,2))*t/2)/(powf(a,2));
        return p;
    }""")

if __name__=="__main__":
    x = -1 * np.random.rand(10).astype(np.float32)
    x = np.linspace(-5,5,10).astype(np.float32)
    print x
    x_gpu = gpuarray.to_gpu(x)
    out_gpu = gpuarray.empty_like(x_gpu)
    print out_gpu

    out = hddm.likelihoods.wfpt.pdf(x, 1., 0., 2, .5, 0., 0., 0., 1e-4)

    pdf_gpu(x_gpu, 1., 2., .5, 0., 1e-4, out_gpu)

    print out
    print np.array(out_gpu.get())

#pdf = pycuda.compiler.SourceModule(kernel_source_wfpt)
#pdf = kernel_source_wfpt.get_function("pdf")
