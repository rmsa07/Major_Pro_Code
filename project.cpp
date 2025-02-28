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


//ok updated
bool canInsertAtPosition(
    const Route& route,  // Pass by reference to avoid copying
    const std::vector<Customer>& customers,
    int customerId, 
    int position, 
    int vehicleCapacity, 
    int depotCloseTime
) {
    // Prevent inserting before depot
    if (position == 0 || position >= route.customers.size()) return false;

    // Check capacity constraint early
    int newTotalDemand = route.totalDemand + customers[customerId].demand;
    if (newTotalDemand > vehicleCapacity) return false;

    // Simulate travel time with insertion
    int currentTime = 0;
    for (size_t i = 1; i < route.customers.size(); ++i) {
        int prevCustomerId = route.customers[i - 1];
        int currentCustomerId = route.customers[i];
        if (i == position) currentCustomerId = customerId;  // Simulate insertion

        const Customer& prevCustomer = customers[prevCustomerId];
        const Customer& currentCustomer = customers[currentCustomerId];

        // Travel time calculation
        int travelTime = calculateTravelTime(prevCustomer, currentCustomer);
        currentTime = std::max(currentTime + travelTime, currentCustomer.readyTime);

        // Check time window constraint
        if (currentTime > currentCustomer.dueTime) return false;

        currentTime += currentCustomer.serviceTime;
    }

    // Check depot return feasibility (only once)
    int lastCustomerId = route.customers.back();
    if (position == route.customers.size()) lastCustomerId = customerId;  // Simulate last position
    int returnTime = currentTime + calculateTravelTime(customers[lastCustomerId], customers[0]);
    
    if (returnTime > customers[0].dueTime) return false;

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

//ok updated
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
Solution crossover(const Solution &parent1, const Solution &parent2, const vector<Customer> &customers, int vehicleCapacity, int depotCloseTime, int objective) {
    Solution child;
    set<int> visitedCustomers;
    vector<Route> childRoutes;

    // Phase 1: Select "k" best routes from both parents to maintain diversity
    int MINR = min(parent1.routes.size(), parent2.routes.size());
    int k = (objective == 1) ? (MINR - 1) : MINR;
    
    vector<Route> candidateRoutes;
    candidateRoutes.insert(candidateRoutes.end(), parent1.routes.begin(), parent1.routes.end());
    candidateRoutes.insert(candidateRoutes.end(), parent2.routes.begin(), parent2.routes.end());

    // Sort routes by objective (either max customers or min distance ratio)
    sort(candidateRoutes.begin(), candidateRoutes.end(), [&](const Route &a, const Route &b) {
        if (objective == 1) return a.customers.size() > b.customers.size();
        else return (a.totalDistance / a.customers.size()) < (b.totalDistance / b.customers.size());
    });

    for (int i = 0; i < k; ++i) {
        childRoutes.push_back(candidateRoutes[i]);
        for (int customerId : candidateRoutes[i].customers) {
            visitedCustomers.insert(customerId);
        }
    }

    // Phase 2: Insert unassigned customers using best-cost insertion
    for (const auto &parent : {parent1, parent2}) {
        for (const auto &route : parent.routes) {
            for (int customerId : route.customers) {
                if (visitedCustomers.find(customerId) == visitedCustomers.end()) {
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

    return evaluateSolution(childRoutes, customers);
}

//ok updated
// Mutation Operator
Solution mutate(const Solution &parent, const vector<Customer> &customers, int vehicleCapacity, int depotCloseTime, int objective) {
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

    // Step 7: Optimize depot visits by removing unnecessary depot stops
    for (auto &route : solution.routes) {
        if (route.customers.size() > 2 && route.customers.front() == 0 && route.customers.back() == 0) {
            route.customers.erase(route.customers.begin());
            route.customers.pop_back();
        }
    }

    // Step 8: Recalculate the total distance & vehicles
    solution = evaluateSolution(solution.routes, customers);
    return solution;
}

//ok updated
// Function to Generate Initial Population
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
            for (size_t position = 1; position <= currentroute.customers.size(); ++position) {
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


// Function to perform non-dominated sorting
vector<vector<Solution>> nonDominatedSorting(vector<Solution> &population) {
    int populationSize = population.size();
    vector<vector<int>> fronts;    // Pareto fronts
    vector<int> dominationCount(populationSize, 0); // Count of solutions dominating each solution
    vector<vector<int>> dominatedSolutions(populationSize); // Solutions dominated by each solution

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


// Multi-Objective Optimization
// std::vector<Solution>
void  optimize(const std::vector<Customer> &customers, int populationSize, int generations, int vehicleCapacity, int depotReadyTime, int depotCloseTime) {
    std::vector<Solution> population;
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
            int idx1 = std::rand() % population.size();
            int idx2 = std::rand() % population.size();

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

        // if(population.size() > 2*populationSize){
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
        int depotCloseTime = 230;

   
    std::vector<Customer> customers = {
{0,35.00,35.00,0.00,0.00,230.00,0.00},
{1,41.00,49.00,10.00,0.00,204.00,10.00},
{2,35.00,17.00,7.00,0.00,202.00,10.00},
{3,55.00,45.00,13.00,0.00,197.00,10.00},
{4,55.00,20.00,19.00,149.00,159.00,10.00},
{5,15.00,30.00,26.00,0.00,199.00,10.00},
{6,25.00,30.00,3.00,99.00,109.00,10.00},
{7,20.00,50.00,5.00,0.00,198.00,10.00},
{8,10.00,43.00,9.00,95.00,105.00,10.00},
{9,55.00,60.00,16.00,97.00,107.00,10.00},
{10,30.00,60.00,16.00,124.00,134.00,10.00},
{11,20.00,65.00,12.00,67.00,77.00,10.00},
{12,50.00,35.00,19.00,0.00,205.00,10.00},
{13,30.00,25.00,23.00,159.00,169.00,10.00},
{14,15.00,10.00,20.00,32.00,42.00,10.00},
{15,30.00,5.00,8.00,61.00,71.00,10.00},
{16,10.00,20.00,19.00,75.00,85.00,10.00},
{17,5.00,30.00,2.00,157.00,167.00,10.00},
{18,20.00,40.00,12.00,87.00,97.00,10.00},
{19,15.00,60.00,17.00,76.00,86.00,10.00},
{20,45.00,65.00,9.00,126.00,136.00,10.00},
{21,45.00,20.00,11.00,0.00,201.00,10.00},
{22,45.00,10.00,18.00,97.00,107.00,10.00},
{23,55.00,5.00,29.00,68.00,78.00,10.00},
{24,65.00,35.00,3.00,153.00,163.00,10.00}
};
  
 
    int populationSize = 20;
    int generations = 100;
    
    optimize(customers, populationSize, generations, vehicleCapacity, depotReadyTime, depotCloseTime);

    return 0;

}