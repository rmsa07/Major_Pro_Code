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





// Helper function to find the minimum and maximum of a vector
template <typename T>
std::pair<T, T> findMinMax(const std::vector<T>& vec) {
    if (vec.empty()) {
        return {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::lowest()};
    }
    return {*std::min_element(vec.begin(), vec.end()), *std::max_element(vec.begin(), vec.end())};
}

// Function to normalize objective values to [0, 1]
void normalizeObjectives(std::vector<Solution>& population) {
    if (population.empty()) return;

    std::vector<double> distances;
    std::vector<int> numVehiclesVec;
    for (const auto& sol : population) {
        distances.push_back(sol.totalDistance);
        numVehiclesVec.push_back(sol.numVehicles);
    }

    std::pair<double, double> minMaxDistance = findMinMax(distances);
    double minDistance = minMaxDistance.first;
    double maxDistance = minMaxDistance.second;

    std::pair<int, int> minMaxVehicles = findMinMax(numVehiclesVec);
    int minVehicles = minMaxVehicles.first;
    int maxVehicles = minMaxVehicles.second;

    double distanceRange = maxDistance - minDistance;
    int vehicleRange = maxVehicles - minVehicles;

    // Avoid division by zero
    if (distanceRange == 0) distanceRange = 1e-9;
    if (vehicleRange == 0) vehicleRange = 1e-9;

    for (auto& sol : population) {
        sol.totalDistance = (sol.totalDistance - minDistance) / distanceRange;
        sol.numVehicles = (sol.numVehicles - minVehicles) / vehicleRange;
    }
}

// Function to generate reference points for a bi-objective problem
std::vector<std::pair<double, double>> generateReferencePoints(int numDivisions) {
    std::vector<std::pair<double, double>> refPoints;
    for (int i = 0; i <= numDivisions; ++i) {
        refPoints.push_back({(double)i / numDivisions, (double)(numDivisions - i) / numDivisions});
    }
    return refPoints;
}

// Function to calculate the Euclidean distance between a solution and a reference point
double euclideanDistance(const Solution& sol, const std::pair<double, double>& refPoint) {
    return std::sqrt(std::pow(sol.totalDistance - refPoint.first, 2) + std::pow(sol.numVehicles - refPoint.second, 2));
}

// Function to associate solutions with the closest reference point
void associateSolutions(const std::vector<Solution>& front, const std::vector<std::pair<double, double>>& refPoints, std::vector<int>& association, std::vector<int>& refPointCounts) {
    association.resize(front.size());
    for (size_t i = 0; i < front.size(); ++i) {
        double minDistance = std::numeric_limits<double>::max();
        int closestRefPointIndex = -1;
        for (size_t j = 0; j < refPoints.size(); ++j) {
            double dist = euclideanDistance(front[i], refPoints[j]);
            if (dist < minDistance) {
                minDistance = dist;
                closestRefPointIndex = j;
            }
        }
        association[i] = closestRefPointIndex;
        if (closestRefPointIndex != -1) {
            refPointCounts[closestRefPointIndex]++;
        }
    }
}

// Function to select individuals based on reference point association and niching
void selectBasedOnReferencePoints(const std::vector<Solution>& lastFront, const std::vector<std::pair<double, double>>& refPoints, const std::vector<int>& association, std::vector<int>& refPointCounts, std::vector<Solution>& nextPopulation, int populationSize) {
    std::vector<int> frontIndices(lastFront.size());
    for (size_t i = 0; i < lastFront.size(); ++i) {
        frontIndices[i] = i;
    }

    while (nextPopulation.size() < populationSize && !frontIndices.empty()) {
        int minCount = std::numeric_limits<int>::max();
        int chosenRefPointIndex = -1;

        // Find the least crowded reference point
        for (size_t i = 0; i < refPoints.size(); ++i) {
            if (refPointCounts[i] < minCount) {
                minCount = refPointCounts[i];
                chosenRefPointIndex = i;
            }
        }

        if (chosenRefPointIndex != -1) {
            std::vector<int> associatedSolutions;
            for (int index : frontIndices) {
                if (association[index] == chosenRefPointIndex) {
                    associatedSolutions.push_back(index);
                }
            }

            if (!associatedSolutions.empty()) {
                // Select one solution associated with this reference point (e.g., randomly)
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> distrib(0, associatedSolutions.size() - 1);
                int selectedSolutionIndexInFront = associatedSolutions[distrib(gen)];
                nextPopulation.push_back(lastFront[selectedSolutionIndexInFront]);

                // Increment the count for the chosen reference point
                refPointCounts[chosenRefPointIndex]++;

                // Remove the selected solution from further consideration
                auto it = std::find(frontIndices.begin(), frontIndices.end(), selectedSolutionIndexInFront);
                if (it != frontIndices.end()) {
                    frontIndices.erase(it);
                }
            } else {
                // If no solution is associated with this least crowded point, move on
                refPointCounts[chosenRefPointIndex] = std::numeric_limits<int>::max(); // To avoid re-selecting
            }
        } else {
            break; // Should not happen if frontIndices is not empty
        }
    }

    // If nextPopulation is still smaller than populationSize, fill it with the remaining of the last front
    while (nextPopulation.size() < populationSize && !frontIndices.empty()) {
        nextPopulation.push_back(lastFront[frontIndices.back()]);
        frontIndices.pop_back();
    }
}




void optimizeNSGAIII(const std::vector<Customer> &customers, int populationSize, int generations, int vehicleCapacity, int depotReadyTime, int depotCloseTime, int numRefPointDivisions) {
    std::vector<Solution> population;
    // ... (initial population generation) ...
    for (int i = 0; i < populationSize; ++i) {
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

    for (int gen = 0; gen < generations; ++gen) {
        std::vector<Solution> combinedPopulation = population;
        std::vector<Solution> offspring;
        for (int i = 0; i < 25; ++i) { // Generate a population-sized offspring set
            int parent1_idx = tournamentSelection(population);
            int parent2_idx = tournamentSelection(population);

            if (parent1_idx != parent2_idx) {
                for(int obj =1;obj<=2;obj++){
                    Solution child = crossover(population[parent1_idx], population[parent2_idx], customers, vehicleCapacity, depotCloseTime, obj); // Decide on the objective parameter
                    Solution mutated_child = mutate(child, customers, vehicleCapacity, depotCloseTime, obj); // Decide on the objective parameter
                    offspring.push_back(mutated_child);
                }   
            } else {
                // Handle the case where parents are the same (e.g., add a copy of the parent or do nothing)
                offspring.push_back(population[parent1_idx]); // Example: add a copy
            }
        }

        // Combine the current population with the offspring
        combinedPopulation.insert(combinedPopulation.end(), offspring.begin(), offspring.end());

        // Now sort the combined population
        auto fronts = nonDominatedSorting(combinedPopulation);
        // for (int i = 0; i < fronts.size(); ++i) {
        // // cout << "Front " << i + 1 << ": ";
        //     cout<<fronts[i].size()<<endl;
        //     for (auto idx : fronts[i]) {
        //         cout << idx.totalDistance << " " << idx.numVehicles<<endl; // Index of solution in population
        //     }
        //     cout <<endl;
        // }

        std::vector<Solution> nextPopulation;
        int frontIndex = 0;
        while (frontIndex < fronts.size() && nextPopulation.size() + fronts[frontIndex].size() <= populationSize) {
            nextPopulation.insert(nextPopulation.end(), fronts[frontIndex].begin(), fronts[frontIndex].end());
            frontIndex++;
        }

        if (nextPopulation.size() < populationSize && frontIndex < fronts.size()) {
            std::vector<Solution>& lastFront = fronts[frontIndex];
            normalizeObjectives(combinedPopulation);
            auto referencePoints = generateReferencePoints(numRefPointDivisions);
            std::vector<int> association(lastFront.size());
            std::vector<int> refPointCounts(referencePoints.size(), 0);
            associateSolutions(lastFront, referencePoints, association, refPointCounts);
            selectBasedOnReferencePoints(lastFront, referencePoints, association, refPointCounts, nextPopulation, populationSize);
        } else {
            // If all non-dominated fronts fit, the next population is just those fronts
            // (or truncated if it exceeds populationSize, but the while loop handles this)
        }

        population = nextPopulation;
        // std::cout << "Generation " << gen + 1 << " - Population size: " << population.size() << std::endl;
    }

    std::cout << "\nFinal Pareto Front (first 10 solutions):" << std::endl;
    // normalizeObjectives(population); // Normalize for final output as well
    auto finalFronts = nonDominatedSorting(population);
     for (int i = 0; i < finalFronts.size(); ++i) {
        // cout << "Front " << i + 1 << ": ";
            cout<<finalFronts[i].size()<<endl;
            for (auto idx : finalFronts[i]) {
                cout << idx.totalDistance << " " << idx.numVehicles<<endl; // Index of solution in population
            }
            cout <<endl;
        }
}


signed main(){
     #ifndef ONLINE_JUDGE
        freopen("input.txt","r",stdin);
        // freopen("output.txt","w",stdout);
      #endif
 

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
{25, 35.00, 5.00, 20.00, 154.00, 184.00, 10.00},
{26, 95.00, 30.00, 30.00, 115.00, 145.00, 10.00},
{27, 95.00, 35.00, 20.00, 62.00, 92.00, 10.00},
{28, 92.00, 30.00, 10.00, 62.00, 92.00, 10.00},
{29, 90.00, 35.00, 10.00, 67.00, 97.00, 10.00},
{30, 88.00, 30.00, 10.00, 74.00, 104.00, 10.00},
{31, 88.00, 35.00, 20.00, 61.00, 91.00, 10.00},
{32, 87.00, 30.00, 10.00, 131.00, 161.00, 10.00},
{33, 85.00, 25.00, 10.00, 51.00, 81.00, 10.00},
{34, 85.00, 35.00, 30.00, 111.00, 141.00, 10.00},
{35, 67.00, 85.00, 20.00, 139.00, 169.00, 10.00},
{36, 65.00, 85.00, 40.00, 43.00, 73.00, 10.00},
{37, 65.00, 82.00, 10.00, 124.00, 154.00, 10.00},
{38, 62.00, 80.00, 30.00, 75.00, 105.00, 10.00},
{39, 60.00, 80.00, 10.00, 37.00, 67.00, 10.00},
{40, 60.00, 85.00, 30.00, 85.00, 115.00, 10.00},
{41, 58.00, 75.00, 20.00, 92.00, 122.00, 10.00},
{42, 55.00, 80.00, 10.00, 33.00, 63.00, 10.00},
{43, 55.00, 85.00, 20.00, 128.00, 158.00, 10.00},
{44, 55.00, 82.00, 10.00, 64.00, 94.00, 10.00},
{45, 20.00, 82.00, 10.00, 37.00, 67.00, 10.00},
{46, 18.00, 80.00, 10.00, 113.00, 143.00, 10.00},
{47, 2.00, 45.00, 10.00, 45.00, 75.00, 10.00},
{48, 42.00, 5.00, 10.00, 151.00, 181.00, 10.00},
{49, 42.00, 12.00, 10.00, 104.00, 134.00, 10.00},
{50, 72.00, 35.00, 30.00, 116.00, 146.00, 10.00},
{51, 55.00, 20.00, 19.00, 83.00, 113.00, 10.00},
{52, 25.00, 30.00, 3.00, 52.00, 82.00, 10.00},
{53, 20.00, 50.00, 5.00, 91.00, 121.00, 10.00},
{54, 55.00, 60.00, 16.00, 139.00, 169.00, 10.00},
{55, 30.00, 60.00, 16.00, 140.00, 170.00, 10.00},
{56, 50.00, 35.00, 19.00, 130.00, 160.00, 10.00},
{57, 30.00, 25.00, 23.00, 96.00, 126.00, 10.00},
{58, 15.00, 10.00, 20.00, 152.00, 182.00, 10.00},
{59, 10.00, 20.00, 19.00, 42.00, 72.00, 10.00},
{60, 15.00, 60.00, 17.00, 155.00, 185.00, 10.00},
{61, 45.00, 65.00, 9.00, 66.00, 96.00, 10.00},
{62, 65.00, 35.00, 3.00, 52.00, 82.00, 10.00},
{63, 65.00, 20.00, 6.00, 39.00, 69.00, 10.00},
{64, 45.00, 30.00, 17.00, 53.00, 83.00, 10.00},
{65, 35.00, 40.00, 16.00, 11.00, 41.00, 10.00},
{66, 41.00, 37.00, 16.00, 133.00, 163.00, 10.00},
{67, 64.00, 42.00, 9.00, 70.00, 100.00, 10.00},
{68, 40.00, 60.00, 21.00, 144.00, 174.00, 10.00},
{69, 31.00, 52.00, 27.00, 41.00, 71.00, 10.00},
{70, 35.00, 69.00, 23.00, 180.00, 210.00, 10.00},
{71, 65.00, 55.00, 14.00, 65.00, 95.00, 10.00},
{72, 63.00, 65.00, 8.00, 30.00, 60.00, 10.00},
{73, 2.00, 60.00, 5.00, 77.00, 107.00, 10.00},
{74, 20.00, 20.00, 8.00, 141.00, 171.00, 10.00},
{75, 5.00, 5.00, 16.00, 74.00, 104.00, 10.00},
{76, 60.00, 12.00, 31.00, 75.00, 105.00, 10.00},
{77, 23.00, 3.00, 7.00, 150.00, 180.00, 10.00},
{78, 8.00, 56.00, 27.00, 90.00, 120.00, 10.00},
{79, 6.00, 68.00, 30.00, 89.00, 119.00, 10.00},
{80, 47.00, 47.00, 13.00, 192.00, 222.00, 10.00},
{81, 49.00, 58.00, 10.00, 86.00, 116.00, 10.00},
{82, 27.00, 43.00, 9.00, 42.00, 72.00, 10.00},
{83, 37.00, 31.00, 14.00, 35.00, 65.00, 10.00},
{84, 57.00, 29.00, 18.00, 96.00, 126.00, 10.00},
{85, 63.00, 23.00, 2.00, 87.00, 117.00, 10.00},
{86, 21.00, 24.00, 28.00, 87.00, 117.00, 10.00},
{87, 12.00, 24.00, 13.00, 90.00, 120.00, 10.00},
{88, 24.00, 58.00, 19.00, 67.00, 97.00, 10.00},
{89, 67.00, 5.00, 25.00, 144.00, 174.00, 10.00},
{90, 37.00, 47.00, 6.00, 86.00, 116.00, 10.00},
{91, 49.00, 42.00, 13.00, 167.00, 197.00, 10.00},
{92, 53.00, 43.00, 14.00, 14.00, 44.00, 10.00},
{93, 61.00, 52.00, 3.00, 178.00, 208.00, 10.00},
{94, 57.00, 48.00, 23.00, 95.00, 125.00, 10.00},
{95, 56.00, 37.00, 6.00, 34.00, 64.00, 10.00},
{96, 55.00, 54.00, 26.00, 132.00, 162.00, 10.00},
{97, 4.00, 18.00, 35.00, 120.00, 150.00, 10.00},
{98, 26.00, 52.00, 9.00, 46.00, 76.00, 10.00},
{99, 26.00, 35.00, 15.00, 77.00, 107.00, 10.00},
{100, 31.00, 67.00, 3.00, 180.00, 210.00, 10.00},





    };


    int populationSize = 30;
    int generations = 1000;
    int vehicleCapacity = 200;
    int depotReadyTime = 0;
    int depotCloseTime = 240;
    int numRefPointDivisions = 3; // Experiment with this value

    // Call the NSGA-III optimization function
    optimizeNSGAIII(customers, populationSize, generations, vehicleCapacity, depotReadyTime, depotCloseTime, numRefPointDivisions);

    return 0;

}