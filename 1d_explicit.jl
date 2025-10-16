using NonlinearSolve

include("parameters.jl")

# beta = 1
# gamma = -1.0
# K = 2.0
# h = 2.0
# c = 1.0
# z = 3.0

root1 = -gamma - sqrt(2*beta+gamma^2) 
root2 = -gamma + sqrt(2*beta+gamma^2)

rootK = -(gamma - drift_b) - sqrt(2*beta + (gamma-drift_b)^2)

g(x, p) = (h/beta*(-root1)*exp(root1*x)-(h/beta-c)*(-rootK))*root2*(exp(root1*x)-exp(root2*x))-(h/beta-c-h/beta*exp(root1*x))*(root1*root2*exp(root1*x)-root2*root2*exp(root2*x))-p

x0 = 10.0
p = 0.0

my_prob = NonlinearProblem(g, x0, p)

drift_sol = solve(my_prob) #this is the nonlinear solver that solves for the critical threshold b; V'(b) = c the pushing cost.

C2 = (h/beta*(-root1)*exp(root1*drift_sol.u)-(h/beta-c)*(-rootK))/(root1*root2*exp(root1*drift_sol.u)-root2*root2*exp(root2*drift_sol.u))
C3 = (h/beta-c)*exp(-drift_sol.u*rootK)/(-rootK)

#println(drift_sol.u, ", ", C2, ", ", C3)

#the code above computes the constants root1, root2, rootK, C2 and C3. 

#function below computes value function
function VF(z, beta, gamma, h, c, b, K, C2, C3)
    if z<=b
        return h/beta*(z-exp(root1*z)/root1)+gamma*h/beta^2-C2*root2/root1*exp(root1*z)+C2*exp(root2*z)
    else
        return h/beta*z+K*c/beta+(gamma-K)*h/beta^2+C3*exp(rootK*z)
    end
end

#function below computes gradient function. b is the critical threshold at which V'(b) = c the pushing cost; think of K as the drift upper bound.

function DVF(z, beta, gamma, h, c, b, K, C2, C3)
    if z<=b
        return h/beta*(1-exp(root1*z)) - C2*root2*exp(root1*z) + C2*root2*exp(root2*z)
    else
        return h/beta+C3*rootK*exp(rootK*z)
    end
end

println("Discount factor = ", beta, "; initial state z = ", z, "; b = ", drift_sol.u, ", V(z) = ", VF(z, beta, gamma, h, c, drift_sol.u, drift_b, C2, C3), ", V'(z) = ", DVF(z, beta, gamma, h, c, drift_sol.u, drift_b, C2, C3), "\n")

#println("Drift control: z = ", z, ", K = ", K, ", b = ", drift_sol.u, ", V(z) = ", VF(z, beta, gamma, h, c, drift_sol.u, K, C2, C3), ", V'(z) = ", DVF(z, beta, gamma, h, c, drift_sol.u, K, C2, C3))
