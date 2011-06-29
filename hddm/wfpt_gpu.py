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

pdf_gpu = ElementwiseKernel(
    "const float *x, const float v, const float V, const float a, const float z, const float ter, float err, float *out",
    "out[i] = pdf(x[i], v, V, a, z, ter, err)",
    "wfpt_gpu",
    preamble=
    """
    #include <math_constants.h>
    __device__ float pdf(const float x, const float v_in, const float V, const float a, const float z_in, const float ter, const float err)
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
        // p *= expf(-v*a*w -(powf(v,2))*t/2)/(powf(a,2));
        p *= expf((powf((a*w*V),2) - 2*a*v*w - (powf(v,2))*t)/(2*(powf(V,2))*t+2))/sqrtf((powf(V,2))*t+1)/(powf(a,2));
        return p;
    }""")

if __name__=="__main__":
    x = -1 * np.random.rand(10).astype(np.float32)
    x = np.linspace(-5,5,10).astype(np.float32)
    x_gpu = gpuarray.to_gpu(x)
    out_gpu = gpuarray.empty_like(x_gpu)

    out = hddm.likelihoods.wfpt.pdf(x, 1., .5, 2, .5, 0., 0., 0., 1e-4)

    pdf_gpu(x_gpu, 1., 0.5, 2., .5, 0., 1e-4, out_gpu)

    print out
    print np.array(out_gpu.get())

#pdf = pycuda.compiler.SourceModule(kernel_source_wfpt)
#pdf = kernel_source_wfpt.get_function("pdf")
