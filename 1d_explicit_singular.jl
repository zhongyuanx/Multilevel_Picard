using NonlinearSolve

beta = 0.2
gamma = -1.0
h = 2.0
c = 1.0
z = 0.7

r1 = -gamma - sqrt(2*beta+gamma^2)
r2 = -gamma + sqrt(2*beta+gamma^2)

g(b, p) = r1*h/beta*exp(r1*b)/(r2*exp(r2*b)-r1*exp(r1*b)) - (c-h/beta*(1-exp(r1*b)))/(exp(r2*b)-exp(r1*b))-p
#(h/beta*(-root1)*exp(root1*x)-(h/beta-c)*(-rootK))*root2*(exp(root1*x)-exp(root2*x))-(h/beta-c-h/beta*exp(root1*x))*(root1*root2*exp(root1*x)-root2*root2*exp(root2*x))-p

b0 = 1.0
p = 0.0

my_prob = NonlinearProblem(g, b0, p)

sol = solve(my_prob)

C2 = (c-h/beta*(1-exp(r1*sol.u)))/(exp(r2*sol.u)-exp(r1*sol.u))/r2

#println(sol.u, ", ", C2)

function VF(z, beta, gamma, h, c, b, C2)
    if z<=b
        return h/beta*(z-exp(r1*z)/r1)+gamma*h/beta^2-C2*r2/r1*exp(r1*z)+C2*exp(r2*z)
    else
        return c*(z-b) + h/beta*(b-exp(r1*b)/r1)+gamma*h/beta^2-C2*r2/r1*exp(r1*b)+C2*exp(r2*b)
    end
end

function DVF(z, beta, gamma, h, c, b, C2)
    if z<=b
        return h/beta*(1-exp(r1*z)) - C2*r2*exp(r1*z) + C2*r2*exp(r2*z)
    else
        return c
    end
end

println("Singular control: z = ", z, ", b = ", sol.u, ", V(z) = ", VF(z, beta, gamma, h, c, sol.u, C2), ", V'(z) = ", DVF(z, beta, gamma, h, c, sol.u, C2))
