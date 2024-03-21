using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class WaypointFollower : MonoBehaviour
{
    public int id = 0;
    public Transform agent;  // Reference to your agent (Capsule)
    public float waypointRadius = 0.1f;
    private int currentWaypointIndex = 0;
    private NavMeshAgent navMeshAgent;

    void Start()
    {
        navMeshAgent = agent.GetComponent<NavMeshAgent>();
        SetNextWaypoint();
    }

    void Update()
    {
        if (navMeshAgent.remainingDistance < waypointRadius && !navMeshAgent.pathPending)
        {
            SetNextWaypoint();
        }
    }

    void SetNextWaypoint()
    {
        if (currentWaypointIndex < transform.parent.GetComponent<CSVReader>().waypoints.Count)
        {
            if (id == 0)
            {
                Vector3 point = transform.parent.GetComponent<CSVReader>().waypoints[currentWaypointIndex];
                navMeshAgent.SetDestination(point);
                currentWaypointIndex++;
            }

            if (id == 1)
            {
                Vector3 point = transform.parent.GetComponent<CSVReader>().waypoints[currentWaypointIndex] + new Vector3(1.1f, 0f, 1.0f);
                navMeshAgent.SetDestination(point);
                currentWaypointIndex++;
            }

            if (id == 2)
            {
                Vector3 point = transform.parent.GetComponent<CSVReader>().waypoints[currentWaypointIndex] + new Vector3(1.2f, 0f, -1.2f);
                navMeshAgent.SetDestination(point);
                currentWaypointIndex++;
            }

            if (id == 3)
            {
                Vector3 point = transform.parent.GetComponent<CSVReader>().waypoints[currentWaypointIndex] + new Vector3(-1.3f, 0f, 1);
                navMeshAgent.SetDestination(point);
                currentWaypointIndex++;
            }

            if (id == 4)
            {
                Vector3 point = transform.parent.GetComponent<CSVReader>().waypoints[currentWaypointIndex] + new Vector3(-1.3f, 0f, -1.1f);
                navMeshAgent.SetDestination(point);
                currentWaypointIndex++;
            }

            //Vector3 point = transform.parent.GetComponent<CSVReader>().waypoints[currentWaypointIndex] + new Vector3(id, 0f, id);
            //navMeshAgent.SetDestination(point);
            //currentWaypointIndex++;
        }
        else
        {
            // All waypoints reached, you may choose to loop or stop here
            Debug.Log("All waypoints reached");
        }
    }
}
