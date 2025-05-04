#include<bits/stdc++.h>
using namespace std;
#define int float

// Structure Definitions
struct Customer {
    int id;
    int x;
    int y;
    int demand;
    int readyTime;
    int dueTime;
    int serviceTime;
};

struct Route {
    std::vector<int> customers;
    int totalDemand = 0;
    double totalDistance = 0;
    int endTime = 0; // Time when route ends
};

// Solution Structure
struct Solution {
    vector<Route> routes;
    double totalDistance = 0;
    int numVehicles = 0;
    int rank = 0; // Pareto front rank
    int index =  0;
    double crowdingDistance = 0.0;
};

// Utility Functions
int calculateTravelTime(const Customer &a, const Customer &b) {
    // return 1;
    // Euclidean distance, can be adjusted for other distances
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}
//ok updated
// Evaluate Solution Fitness
//correst
Solution evaluateSolution(const std::vector<Route> &routes, const std::vector<Customer> &customers) {
    Solution solution;
    solution.numVehicles = routes.size();
    solution.totalDistance = 0.0; // Ensure proper initialization

    for (const auto &route : routes) {
        if (route.customers.empty()) continue; // Skip empty routes

        double routeDistance = 0;
        int prevCustomer = 0; // Start at depot

        for (size_t i = 0; i < route.customers.size(); ++i) {
            int customerId = route.customers[i];
            routeDistance += calculateTravelTime(customers[prevCustomer], customers[customerId]);
            prevCustomer = customerId;
        }

        // Return to depot
        routeDistance += calculateTravelTime(customers[prevCustomer], customers[0]);

        solution.totalDistance += routeDistance;
    }

    solution.routes = routes;
    return solution;
}
// correct
void evaluateSolution(Solution &solution, const std::vector<Customer> &customers) {
    // Solution solution;
    solution.numVehicles = solution.routes.size();
    solution.totalDistance = 0.0; // Ensure proper initialization

    for (const auto &route : solution.routes) {
        if (route.customers.empty()) continue; // Skip empty routes

        double routeDistance = 0;
        int prevCustomer = 0; // Start at depot

        for (size_t i = 0; i < route.customers.size(); ++i) {
            int customerId = route.customers[i];
            routeDistance += calculateTravelTime(customers[prevCustomer], customers[customerId]);
            prevCustomer = customerId;
        }

        // Return to depot
        routeDistance += calculateTravelTime(customers[prevCustomer], customers[0]);

        solution.totalDistance += routeDistance;
    }

}


//ok updated
//correct
bool canInsertAtPosition(
    const Route& route,
    const std::vector<Customer>& customers,
    int customerId,
    int position,
    int vehicleCapacity,
    int depotCloseTime
) {
    // Prevent insertion before the depot or beyond the end of the route.
    if (position == 0 || position >= route.customers.size()) {
        return false;
    }

    // Create a temporary route to test the insertion without modifying the original.
    Route tempRoute = route;
    tempRoute.customers.insert(tempRoute.customers.begin() + position, customerId);
    tempRoute.totalDemand += customers[customerId].demand;

    // Check if the insertion violates the vehicle capacity constraint.
    if (tempRoute.totalDemand > vehicleCapacity) {
        return false;
    }

    // Initialize the current time with the depot's ready time.
    double currentTime = customers[0].readyTime;

    // Iterate through the modified route to check time window feasibility.
    for (size_t i = 1; i < tempRoute.customers.size(); ++i) {
        int prevCustomerId = tempRoute.customers[i - 1];
        int currentCustomerId = tempRoute.customers[i];

        const Customer& prevCustomer = customers[prevCustomerId];
        const Customer& currentCustomer = customers[currentCustomerId];

        // Calculate the travel time between the previous and current customers.
        double travelTime = calculateTravelTime(prevCustomer, currentCustomer);

        // Update the current time, ensuring it's not before the customer's ready time.
        currentTime = std::max(currentTime + travelTime, (double)currentCustomer.readyTime);

        // Check if the current time exceeds the customer's due time (time window violation).
        if (currentTime > currentCustomer.dueTime) {
            return false;
        }

        // Add the service time at the current customer.
        currentTime += currentCustomer.serviceTime;
    }

    // Get the ID of the last customer in the modified route.
    int lastCustomerId = tempRoute.customers.back();

    // Calculate the return time to the depot.
    double returnTime = currentTime + calculateTravelTime(customers[lastCustomerId], customers[0]);

    // Check if the return time violates the depot's time window constraints.
    if (returnTime > customers[0].dueTime || returnTime < customers[0].readyTime) {
        return false;
    }

    // If all constraints are satisfied, the insertion is feasible.
    return true;
}


// small functionality later

Solution Better(const Solution &parent1, const Solution &parent2,int obj){
    if(obj == 1){ // vehicle min
        if(parent1.numVehicles <= parent2.numVehicles)return parent1;
        return parent2;
    }
    if(parent1.totalDistance <= parent2.totalDistance)return parent1;
    return parent2;
}
bool isdominate(const Solution &parent1, const Solution &parent2){
    if(parent1.numVehicles<parent2.numVehicles && parent1.totalDistance<parent2.totalDistance)
        return true;
    return false;
}
bool is_good (const Solution &parent1, const Solution &parent2,int obj){
    if(obj == 1 && parent1.numVehicles<parent2.numVehicles )return true;
    if(obj == 2 && parent1.totalDistance<parent2.totalDistance)return true;
    return false;
}
bool is_unique_child(const Solution child , const vector<Solution>&population){
     for(auto it:population){
        if(it.numVehicles == child.numVehicles && it.totalDistance == child.totalDistance)return false;
     }
     return true;
}

//ok updated/
//correct
void calculateCrowdingDistance(vector<Solution> &front) {
    int frontSize = front.size();
    if (frontSize <= 2) {
        // Assign infinite crowding distance for trivial fronts
        for (auto &solution : front) {
            solution.crowdingDistance = numeric_limits<double>::infinity();
        }
        return;
    }

    // Step 1: Initialize crowding distance to 0
    for (auto &solution : front) {
        solution.crowdingDistance = 0.0;
    }

    // Step 2: Calculate crowding distance for Objective 1 (totalDistance)
    vector<int> indices(frontSize);
    iota(indices.begin(), indices.end(), 0); // Fill indices [0, 1, 2, ..., frontSize-1]

    sort(indices.begin(), indices.end(), [&](int a, int b) {
        return front[a].totalDistance < front[b].totalDistance;
    });

    // Boundary solutions get infinite crowding distance
    front[indices[0]].crowdingDistance = numeric_limits<double>::infinity();
    front[indices[frontSize - 1]].crowdingDistance = numeric_limits<double>::infinity();

    double minDist = front[indices[0]].totalDistance;
    double maxDist = front[indices[frontSize - 1]].totalDistance;
    double normalized = maxDist - minDist;

    if (normalized > 0.0) {
        for (int i = 1; i < frontSize - 1; ++i) {
            front[indices[i]].crowdingDistance += 
                (front[indices[i + 1]].totalDistance - front[indices[i - 1]].totalDistance) / normalized;
        }
    }

    // Step 3: Calculate crowding distance for Objective 2 (numVehicles)
    sort(indices.begin(), indices.end(), [&](int a, int b) {
        return front[a].numVehicles < front[b].numVehicles;
    });

    // Ensure boundary solutions remain infinite (don't overwrite from previous step)
    front[indices[0]].crowdingDistance = numeric_limits<double>::infinity();
    front[indices[frontSize - 1]].crowdingDistance = numeric_limits<double>::infinity();

    double minVehicles = front[indices[0]].numVehicles;
    double maxVehicles = front[indices[frontSize - 1]].numVehicles;
    double normalizedVehicles = maxVehicles - minVehicles;

    if (normalizedVehicles > 0.0) {
        for (int i = 1; i < frontSize - 1; ++i) {
            front[indices[i]].crowdingDistance += 
                (front[indices[i + 1]].numVehicles - front[indices[i - 1]].numVehicles) / normalizedVehicles;
        }
    }
}


//ok updated
//correct
Solution crossover( Solution parent1,  Solution parent2, const vector<Customer> &customers, int vehicleCapacity, int depotCloseTime, int objective) {
   

    Solution child;
    set<int> visitedCustomers;  // Track assigned customers
    vector<Route> childRoutes;
    
    // Step 1: Store parent routes in vectors
    vector<Route> parentA_routes = parent1.routes;
    vector<Route> parentB_routes = parent2.routes;

    // Determine k based on objective
    int MINR = min(parentA_routes.size(), parentB_routes.size());
    int k = (objective == 1) ? (MINR - 1) : MINR;

    // Step 2: Select k routes iteratively
    for (int i = 0; i < k; ++i) {
        // **Randomly choose a parent**
        bool chooseA = (rand() % 2 == 0);
        vector<Route>& selectedParent = chooseA ? parentA_routes : parentB_routes;
        vector<Route>& otherParent = chooseA ? parentB_routes : parentA_routes;

        // If selected parent has no more routes, switch to the other
        if (selectedParent.empty()) {
            if (!otherParent.empty()) swap(selectedParent, otherParent);
            else break;  // No feasible routes left
        }

        // **Find the most promising route**
        int bestRouteIdx = -1;
        if (objective == 1) {  // f1: Minimize vehicles (Max customers)
            for (size_t j = 0; j < selectedParent.size(); ++j) {
                if (bestRouteIdx == -1 || selectedParent[j].customers.size() > selectedParent[bestRouteIdx].customers.size()) {
                    bestRouteIdx = j;
                }
            }
        } else if (objective == 2) {  // f2: Minimize distance
            for (size_t j = 0; j < selectedParent.size(); ++j) {
                double ratio = selectedParent[j].totalDistance / selectedParent[j].customers.size();
                if (bestRouteIdx == -1 || ratio < (selectedParent[bestRouteIdx].totalDistance / selectedParent[bestRouteIdx].customers.size())) {
                    bestRouteIdx = j;
                }
            }
        }

        if (bestRouteIdx == -1) break;  // No feasible route found

        // **Step 2.2: Add best route to child**
        Route bestRoute = selectedParent[bestRouteIdx];
        childRoutes.push_back(bestRoute);
        
        // **Mark customers as assigned**
        for (int customerId : bestRoute.customers) {
            if (customerId != 0) visitedCustomers.insert(customerId);  // Ignore depot
        }

        // **Step 2.3: Remove selected route from the parent**
        selectedParent.erase(selectedParent.begin() + bestRouteIdx);

        // **Step 2.4: Remove customers from the other parent**
        for (auto &route : otherParent) {
            vector<int> newCustomers = {0};  // Always start with depot
            for (size_t j = 1; j < route.customers.size() - 1; ++j) {  // Ignore first & last (depot)
                if (!visitedCustomers.count(route.customers[j])) {
                    newCustomers.push_back(route.customers[j]);
                }
            }
            newCustomers.push_back(0);  // Always end with depot

            if (newCustomers.size() > 2) {  // Ensure it's still a valid route
                route.customers = newCustomers;
            } else {
                route.customers.clear();  // Mark for removal
            }
        }

        // Remove empty routes
        otherParent.erase(remove_if(otherParent.begin(), otherParent.end(), 
            [](const Route& r) { return r.customers.empty(); }), otherParent.end());
        evaluateSolution(selectedParent,customers);
        evaluateSolution(otherParent,customers);
    }


 
    // Phase 2: Insert unassigned customers using best-cost insertion
    for (const auto &parent : {parent1, parent2}) {
        for (const auto &route : parent.routes) {
            for (int customerId : route.customers) {
                if (customerId && visitedCustomers.find(customerId) == visitedCustomers.end() && customerId >0) {
                    // cout<<customerId<<" ";
                    double bestCost = DBL_MAX;
                    Route *bestRoute = nullptr;
                    int bestPos = -1;

                    for (auto &childRoute : childRoutes) {
                        for (size_t pos = 1; pos < childRoute.customers.size(); ++pos) {
                            Route tempRoute = childRoute;
                            tempRoute.customers.insert(tempRoute.customers.begin() + pos, customerId);
                            tempRoute.totalDemand += customers[customerId].demand;

                            if (tempRoute.totalDemand > vehicleCapacity) continue;

                            // Check travel feasibility
                            int currentTime = 0;
                            bool feasible = true;
                            for (size_t i = 1; i < tempRoute.customers.size(); ++i) {
                                int prevCustomer = tempRoute.customers[i - 1];
                                int currCustomer = tempRoute.customers[i];
                                const Customer &prev = customers[prevCustomer];
                                const Customer &curr = customers[currCustomer];

                                currentTime = max(currentTime + calculateTravelTime(prev, curr), curr.readyTime);
                                if (currentTime > curr.dueTime) {
                                    feasible = false;
                                    break;
                                }
                                currentTime += curr.serviceTime;
                            }

                            if (!feasible) continue;

                            // Evaluate cost
                            double newDistance = evaluateSolution({tempRoute}, customers).totalDistance;
                            double oldDistance = evaluateSolution({childRoute}, customers).totalDistance;
                            double cost = newDistance - oldDistance;

                            // Prefer positions that increase crowding distance in NSGA-II
                            if (cost < bestCost) {
                                bestCost = cost;
                                bestRoute = &childRoute;
                                bestPos = pos;
                            }
                        }
                    }

                    // Apply best insertion
                    if (bestRoute) {
                        bestRoute->customers.insert(bestRoute->customers.begin() + bestPos, customerId);
                        bestRoute->totalDemand += customers[customerId].demand;
                        visitedCustomers.insert(customerId);
                    } else {
                        // Create a new route if no feasible position is found
                        Route newRoute;
                        newRoute.customers.push_back(0);
                        newRoute.customers.push_back(customerId);
                        newRoute.customers.push_back(0);
                        newRoute.totalDemand = customers[customerId].demand;
                        childRoutes.push_back(newRoute);
                        visitedCustomers.insert(customerId);
                    }
                }
            }
        }
    }
     // for(auto it:childRoutes){
     //    for(auto itt:it.customers){
     //        cout<<itt<<" ";
     //    }
     //    cout<<"\n";
     // }
    return evaluateSolution(childRoutes, customers);
}

//ok updated
// Mutation Operator
//correct
Solution mutate(const Solution parent, const vector<Customer> &customers, int vehicleCapacity, int depotCloseTime, int objective) {
    Solution solution = parent;
    if (solution.routes.empty()) return solution;

    // Step 1: Select a random customer from a random route
    int routeIdx = rand() % solution.routes.size();
    Route &selectedRoute = solution.routes[routeIdx];

    if (selectedRoute.customers.size() <= 2) return solution; // Skip if route only has depot

    int customerIdx = 1 + rand() % (selectedRoute.customers.size() - 2);
    int customerId = selectedRoute.customers[customerIdx];

    // Remove customer from current route
    selectedRoute.customers.erase(selectedRoute.customers.begin() + customerIdx);
    selectedRoute.totalDemand -= customers[customerId].demand;

    // Step 2: Evaluate all possible insertions globally
    vector<pair<int, int>> feasibleInsertions; // (routeIdx, position)
    double bestCost = numeric_limits<double>::max();
    bool inserted = false;
    int bestRouteIdx = -1, bestPosition = -1;

    for (size_t r = 0; r < solution.routes.size(); ++r) {
        for (size_t pos = 1; pos < solution.routes[r].customers.size(); ++pos) {
            Route tempRoute = solution.routes[r];
            tempRoute.customers.insert(tempRoute.customers.begin() + pos, customerId);
            tempRoute.totalDemand += customers[customerId].demand;

            // Check feasibility
            if (tempRoute.totalDemand > vehicleCapacity) continue;
            
            // Calculate travel time and feasibility
            int currentTime = 0;
            bool feasible = true;
            for (size_t i = 1; i < tempRoute.customers.size(); ++i) {
                int prevCustomer = tempRoute.customers[i - 1];
                int currCustomer = tempRoute.customers[i];
                const Customer &prev = customers[prevCustomer];
                const Customer &curr = customers[currCustomer];

                currentTime = max(currentTime + calculateTravelTime(prev, curr), curr.readyTime);
                if (currentTime > curr.dueTime) {
                    feasible = false;
                    break;
                }
                currentTime += curr.serviceTime;
            }

            if (!feasible) continue;

            // Step 3: Evaluate the cost of this insertion
            double insertionCost = evaluateSolution({tempRoute}, customers).totalDistance;
            if (objective == 2) { // If minimizing distance
                insertionCost -= evaluateSolution({solution.routes[r]}, customers).totalDistance;
            }

            // Step 4: Store best insertion globally
            if (insertionCost < bestCost) {
                bestCost = insertionCost;
                bestRouteIdx = r;
                bestPosition = pos;
                inserted = true;
            }
        }
    }

    // Step 5: Apply the best insertion found
    if (inserted) {
        solution.routes[bestRouteIdx].customers.insert(solution.routes[bestRouteIdx].customers.begin() + bestPosition, customerId);
        solution.routes[bestRouteIdx].totalDemand += customers[customerId].demand;
    } else {
        // Step 6: If no feasible position found, create a new route
        Route newRoute;
        newRoute.customers.push_back(0); // Depot start
        newRoute.customers.push_back(customerId);
        newRoute.customers.push_back(0); // Depot end
        newRoute.totalDemand = customers[customerId].demand;
        solution.routes.push_back(newRoute);
    }

    // for (auto &route : solution.routes) {
    //     if (route.customers.size() > 2 && route.customers.front() == 0 && route.customers.back() == 0) {
    //         route.customers.erase(route.customers.begin());
    //         route.customers.pop_back();
    //     }
    // }

    // Step 7: Recalculate the total distance & vehicles
    solution = evaluateSolution(solution.routes, customers);
    return solution;
}

//ok updated
// Function to Generate Initial Population
//correct
std::vector<Route> generateInitialPopulation(
    const std::vector<Customer> &customers,
    int vehicleCapacity,
    int depotReadyTime,
    int depotCloseTime
) {
    std::vector<Route> population;
    std::vector<int> unassignedCustomers;

    // Initialize unassigned customers (skip depot)
    for (const auto &customer : customers) {
        if (customer.id != 0) unassignedCustomers.push_back(customer.id);
    }

    // Seed the random generator once in `main()`, not here

    // Process unassigned customers
    while (!unassignedCustomers.empty()) {
        int randomIndex = std::rand() % unassignedCustomers.size();
        int customerId = unassignedCustomers[randomIndex];
        bool assigned = false;

        // Try inserting into existing routes
        for (size_t i = 0; i < population.size(); i++) {
            Route &currentroute = population[i];  // Use reference to modify directly
            for (size_t position = 1; position < currentroute.customers.size(); ++position) {
                if (canInsertAtPosition(currentroute, customers, customerId, position, vehicleCapacity, depotCloseTime)) {
                    currentroute.customers.insert(currentroute.customers.begin() + position, customerId);
                    currentroute.totalDemand += customers[customerId].demand;
                    assigned = true;
                    break;
                }
            }
            if (assigned) break;
        }

        // If no feasible insertion, create a new route (with depot check)
        if (!assigned) {
            Route newRoute;
            newRoute.customers.push_back(0); // Start at depot
            newRoute.totalDemand = customers[customerId].demand;
            newRoute.customers.push_back(customerId);
            
            // Calculate end time
            newRoute.endTime = depotReadyTime + std::max(customers[customerId].readyTime, calculateTravelTime(customers[0], customers[customerId]));
            newRoute.endTime += customers[customerId].serviceTime;
            
            // Check if depot closing time is violated
            if (newRoute.endTime + calculateTravelTime(customers[customerId], customers[0]) <= depotCloseTime) {
                newRoute.customers.push_back(0); // Return to depot
                population.push_back(newRoute);
            } 
        }

        // Remove customer from unassigned list
        unassignedCustomers.erase(unassignedCustomers.begin() + randomIndex);
    }

    return population;
}


// Function to check if a route is feasible
bool isRouteFeasible(const Route& route, const std::vector<Customer>& customers, int vehicleCapacity, int depotCloseTime) {
    if (route.totalDemand > vehicleCapacity) return false;

    int currentTime = 0;
    for (size_t i = 1; i < route.customers.size(); ++i) {
        int prevCustomer = route.customers[i - 1];
        int currCustomer = route.customers[i];
        const Customer &prev = customers[prevCustomer];
        const Customer &curr = customers[currCustomer];

        currentTime = std::max(currentTime + calculateTravelTime(prev, curr), curr.readyTime);
        if (currentTime > curr.dueTime) return false;
        currentTime += curr.serviceTime;
    }

    // Check return to depot
    int lastCustomer = route.customers[route.customers.size() - 2];
    currentTime += calculateTravelTime(customers[lastCustomer], customers[0]);

    if (currentTime > depotCloseTime) return false;

    return true;
}

// Function to calculate the insertion cost (distance difference)
double calculateInsertionCost(const Route& oldRoute, const Route& newRoute, const std::vector<Customer>& customers) {
    double oldDistance = 0.0;
    for (size_t i = 1; i < oldRoute.customers.size(); ++i) {
        oldDistance += calculateTravelTime(customers[oldRoute.customers[i - 1]], customers[oldRoute.customers[i]]);
    }

    double newDistance = 0.0;
    for (size_t i = 1; i < newRoute.customers.size(); ++i) {
        newDistance += calculateTravelTime(customers[newRoute.customers[i - 1]], customers[newRoute.customers[i]]);
    }

    return newDistance - oldDistance;
}
std::vector<Route> generateGreedyPopulation(
    const std::vector<Customer>& customers,
    int vehicleCapacity,
    int depotReadyTime,
    int depotCloseTime
) {
    std::vector<Route> population;
    std::vector<bool> visited(customers.size(), false);
    visited[0] = true; // depot

    while (true) {
        // Create a new route
        Route route;
        route.customers.push_back(0); // start at depot
        route.totalDemand = 0;

        double currentTime = depotReadyTime;
        int currentLocation = 0;

        bool addedAny = false;

        while (true) {
            int nextCustomer = -1;
            double bestTime = 1e9;

            for (size_t i = 1; i < customers.size(); ++i) {
                if (visited[i]) continue;

                const Customer& candidate = customers[i];

                // Capacity check
                if (route.totalDemand + candidate.demand > vehicleCapacity)
                    continue;

                double travelTime = calculateTravelTime(customers[currentLocation], candidate);
                double arrivalTime = std::max(currentTime + travelTime, (double)candidate.readyTime);

                // Time window check
                if (arrivalTime > candidate.dueTime)
                    continue;

                // Check if after serving, we can return to depot in time
                double finishService = arrivalTime + candidate.serviceTime;
                double returnTime = finishService + calculateTravelTime(candidate, customers[0]);

                if (returnTime > depotCloseTime)
                    continue;

                // Select the best (earliest feasible) customer
                if (arrivalTime < bestTime) {
                    bestTime = arrivalTime;
                    nextCustomer = i;
                }
            }

            if (nextCustomer == -1) break; // no more feasible insertions

            // Insert best customer
            route.customers.push_back(nextCustomer);
            visited[nextCustomer] = true;
            addedAny = true;

            // Update route state
            const Customer& cust = customers[nextCustomer];
            double travelTime = calculateTravelTime(customers[currentLocation], cust);
            currentTime = std::max(currentTime + travelTime, (double)cust.readyTime) + cust.serviceTime;
            route.totalDemand += cust.demand;
            currentLocation = nextCustomer;
        }

        if (!addedAny) break; // all customers assigned

        route.customers.push_back(0); // return to depot
        population.push_back(route);
    }

    return population;
}





vector<Solution> unique_population(vector<Solution> &population){
    vector<Solution>unique_populations;
    set<pair<int,int>>seen;
    for(auto it:population){
        if(seen.find({it.totalDistance,it.numVehicles}) == seen.end()){
            unique_populations.push_back(it);
            seen.insert({it.totalDistance,it.numVehicles});
        }
    }
    return unique_populations;
}

// Function to perform non-dominated sorting
// correct
vector<vector<Solution>> nonDominatedSorting(vector<Solution> &population) {
    population = unique_population(population);
    int populationSize = population.size();
    vector<vector<int>> fronts;    // Pareto fronts
    vector<int> dominationCount(populationSize, 0); // Count of solutions dominating each solution
    vector<vector<int>> dominatedSolutions(populationSize); // Solutions dominated by each solution

    for (auto &sol : population) {
    sol.rank = -1;  // Ensure every solution starts unranked
    }

    // Step 1: Compare every pair of solutions
    for (int i = 0; i < populationSize; ++i) {
        for (int j = 0; j < populationSize; ++j) {
            if (i == j) continue;

            // Check if solution[i] dominates solution[j]
            bool dominates = ((population[i].totalDistance < population[j].totalDistance && 
                              population[i].numVehicles <= population[j].numVehicles) ||
                             (population[i].totalDistance <= population[j].totalDistance && 
                              population[i].numVehicles < population[j].numVehicles));

            if (dominates) {
                dominatedSolutions[i].push_back(j); // i dominates j
            } else if ((population[j].totalDistance < population[i].totalDistance && 
                        population[j].numVehicles <= population[i].numVehicles) ||
                       (population[j].totalDistance <= population[i].totalDistance && 
                        population[j].numVehicles < population[i].numVehicles)) {
                dominationCount[i]++; // j dominates i
            } 
        }

        // If no solution dominates i, it belongs to the first front
        if (dominationCount[i] == 0) {
            population[i].rank = 0; // First Pareto front
            if (fronts.empty()) fronts.emplace_back();
            fronts[0].push_back(i);
        }
    }

    // Step 2: Build remaining fronts
    int currentFront = 0;
    while (currentFront < fronts.size()) {
        vector<int> nextFront;

        for (int i : fronts[currentFront]) {
            for (int j : dominatedSolutions[i]) {
                dominationCount[j]--;

                if (dominationCount[j] == 0) {
                    population[j].rank = currentFront + 1; // Assign next rank
                    nextFront.push_back(j);
                }
            }
        }

        if (!nextFront.empty()) fronts.push_back(nextFront);
        currentFront++;
    }

    vector<vector<Solution>>fronts_sol(fronts.size());
    for (int i = 0; i < fronts.size(); ++i) {
        for (int idx : fronts[i]) {
            fronts_sol[i].push_back(population[idx]);
        }
    }


    return fronts_sol;
}

//correct
int tournamentSelection(const std::vector<Solution>& population) {
    int idx1 = std::rand() % population.size();
    int idx2 = std::rand() % population.size();
    
    // Select based on rank (lower is better)
    if (population[idx1].rank < population[idx2].rank) {
        return idx1;
    } else if (population[idx1].rank > population[idx2].rank) {
        return idx2;
    }

    // If rank is the same, use crowding distance (higher is better)
    return (population[idx1].crowdingDistance > population[idx2].crowdingDistance) ? idx1 : idx2;
}


// Multi-Objective Optimization
// std::vector<Solution>
void  optimize(const std::vector<Customer> &customers, int populationSize, int generations, int vehicleCapacity, int depotReadyTime, int depotCloseTime) {
    std::vector<Solution> population;
    for (int i = 0; i < populationSize/2; ++i) {
        auto routes = generateInitialPopulation(customers, vehicleCapacity, depotReadyTime, depotCloseTime);
        // for(auto it:routes){
        //  for(auto itt:it.customers){
        //      cout<<itt<<" ";
        //  }cout<<"\n";
        // }
        // cout<<"neww\n";
        population.push_back(evaluateSolution(routes, customers));
        // cout<<population[i].numVehicles<<" "<<population[i].totalDistance<<"\n";
        // cout<<"\n";

    }
    for (int i = 0; i < populationSize/2; ++i) {
        auto routes = generateGreedyPopulation(customers, vehicleCapacity, depotReadyTime, depotCloseTime);
        // for(auto it:routes){
        //  for(auto itt:it.customers){
        //      cout<<itt<<" ";
        //  }cout<<"\n";
        // }
        // cout<<"neww\n";
        population.push_back(evaluateSolution(routes, customers));
        // cout<<population[i].numVehicles<<" "<<population[i].totalDistance<<"\n";
        // cout<<"\n";

    }

   
    // return ; 
    // non-dominant sorting
    auto fronts = nonDominatedSorting(population);
 
    int N = 0 ;
    int Ng = generations ;

    while (N < Ng){
        bool flag = true ;
        if(4*N > 3*Ng ){
            flag = false;
        }

        for(int i=0;i<25;i++){
            int idx1 = tournamentSelection(population);
            int idx2 = tournamentSelection(population);

            if(idx1 != idx2){
                for(int obj =1;obj<=2;obj++){
                    Solution child , mutate_child ; 
                    if(flag == true){
                         child=crossover(population[idx1],population[idx2],customers,vehicleCapacity,depotCloseTime,obj);
                         mutate_child = mutate(child,customers,vehicleCapacity,depotCloseTime,obj);
                    }else{
                         child = Better(population[idx1],population[idx2],obj);
                         mutate_child = mutate(child,customers,vehicleCapacity,depotCloseTime,obj);
                    }
                    
                    Solution new_child;
                    if(isdominate(mutate_child,child)){
                        new_child = mutate_child;
                    }else if(isdominate(child,mutate_child)){
                        new_child = child;
                    }else{
                        if(is_good(mutate_child,child,obj)){
                           new_child = mutate_child;
                        }else{
                           new_child = child;
                        }
                    }
                    if(is_unique_child(new_child,population)){
                        population.push_back(new_child);
                    }
                    
                }
            }
        }
          
        // if(population.size() > 3*populationSize){
        // cout<<population.size()<<" ";
        // Step 3: Non-Dominated Sorting
        auto fronts = nonDominatedSorting(population);

        // Step 4: Calculate Crowding Distance for Each Front
        for (auto &front : fronts) {
            calculateCrowdingDistance(front);
        }

        // Step 5: Selection for Next Generation
        std::vector<Solution> nextGeneration;
        for ( auto &front : fronts) {
            if (nextGeneration.size() + front.size() <= populationSize) {
                // Add entire front
                for ( auto &solution : front) {
                    nextGeneration.push_back(solution);
                }
            } else {
                // Sort the front by crowding distance (descending)
                vector<int> sortedFront(front.size());
                for (int i = 0; i < front.size(); ++i) {
                    sortedFront[i] = i;
                }

                // Sort by crowding distance
                sort(sortedFront.begin(), sortedFront.end(), [&](int a, int b) {
                    return front[a].crowdingDistance > front[b].crowdingDistance;
                });

                // Add the required solutions to fill the population
                int required =  populationSize - nextGeneration.size();
                for (int i = 0; i < required; ++i) {
                    nextGeneration.push_back(front[sortedFront[i]]);
                }
                break;
            }
        }
        
        population = nextGeneration;
     // }
        
        N++;

    }
    
    // cout<<population.size()<<" ";
    // non-dominant sorting
    fronts = nonDominatedSorting(population);
    cout<<fronts.size()<<endl;
    for (int i = 0; i < fronts.size(); ++i) {
        // cout << "Front " << i + 1 << ": ";
        cout<<fronts[i].size()<<endl;
        for (auto idx : fronts[i]) {
            cout << idx.totalDistance << " " << idx.numVehicles<<endl; // Index of solution in population

            // cout<<idx.routes.size()<<endl;

            // for(auto route : idx.routes){
            //     cout<<route.customers.size()<<endl;
            //     for(auto customer : route.customers){
            //         cout<<customer<<" ";
            //     }
            //     cout<<endl;
            // }

        }
        cout <<endl;
    }
    

}

signed main(){
     #ifndef ONLINE_JUDGE
        freopen("input.txt","r",stdin);
        // freopen("output.txt","w",stdout);
      #endif
 
        int vehicleCapacity = 200;
        int depotReadyTime = 0;
        int depotCloseTime = 240;

   // std::srand(std::time(0));
   //R102 - 50 
    std::vector<Customer> customers = 
   {
{0, 40.00, 50.00, 0.00, 0.00, 240.00, 0.00},  
{1, 25.00, 85.00, 20.00, 145.00, 175.00, 10.00},  
{2, 22.00, 75.00, 30.00, 50.00, 80.00, 10.00},  
{3, 22.00, 85.00, 10.00, 109.00, 139.00, 10.00},  
{4, 20.00, 80.00, 40.00, 141.00, 171.00, 10.00},  
{5, 20.00, 85.00, 20.00, 41.00, 71.00, 10.00},  
{6, 18.00, 75.00, 20.00, 95.00, 125.00, 10.00},  

{7, 15.00, 75.00, 20.00, 79.00, 109.00, 10.00},  
{8, 15.00, 80.00, 10.00, 91.00, 121.00, 10.00},  
{9, 10.00, 35.00, 20.00, 91.00, 121.00, 10.00},  
{10, 10.00, 40.00, 30.00, 119.00, 149.00, 10.00},  
{11, 8.00, 40.00, 40.00, 59.00, 89.00, 10.00},  
{12, 8.00, 45.00, 20.00, 64.00, 94.00, 10.00},  
{13, 5.00, 35.00, 10.00, 142.00, 172.00, 10.00},  

{14, 5.00, 45.00, 10.00, 35.00, 65.00, 10.00},  
{15, 2.00, 40.00, 20.00, 58.00, 88.00, 10.00},  
{16, 0.00, 40.00, 20.00, 72.00, 102.00, 10.00},  
{17, 0.00, 45.00, 20.00, 149.00, 179.00, 10.00},  
{18, 44.00, 5.00, 20.00, 87.00, 117.00, 10.00},  
{19, 42.00, 10.00, 40.00, 72.00, 102.00, 10.00},  
{20, 42.00, 15.00, 10.00, 122.00, 152.00, 10.00},  

{21, 40.00, 5.00, 10.00, 67.00, 97.00, 10.00},  
{22, 40.00, 15.00, 40.00, 92.00, 122.00, 10.00},  
{23, 38.00, 5.00, 30.00, 65.00, 95.00, 10.00},  
{24, 38.00, 15.00, 10.00, 148.00, 178.00, 10.00},  
{25, 35.00, 5.00, 20.00, 154.00, 184.00, 10.00}  
  




    };






 
 
    int populationSize = 30;
    int generations = 1000;
    
    optimize(customers, populationSize, generations, vehicleCapacity, depotReadyTime, depotCloseTime);

    return 0;

}