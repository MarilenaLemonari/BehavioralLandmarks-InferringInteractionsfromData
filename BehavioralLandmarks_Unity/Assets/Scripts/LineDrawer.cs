using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class LineDrawer : MonoBehaviour
{
    public GameObject agentPrefab;
    public float linePointSpacing = 0.1f;

    private LineRenderer lineRenderer;
    private List<Vector3> waypoints = new List<Vector3>();
    private GameObject currentAgent;

    void Start()
    {
        lineRenderer = GetComponent<LineRenderer>();
        lineRenderer.positionCount = 0;
    }

    void Update()
    {
        if (Input.GetMouseButton(0))
        {
            Vector3 mousePosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            mousePosition.z = 0f;

            waypoints.Add(mousePosition);
            DrawLine();

            if (currentAgent != null)
            {
                currentAgent.transform.position = mousePosition;
            }
        }

        if (Input.GetKeyDown(KeyCode.Space) && waypoints.Count > 1)
        {
            MoveAgentAlongPath();
        }
    }

    void DrawLine()
    {
        lineRenderer.positionCount = waypoints.Count;
        lineRenderer.SetPositions(waypoints.ToArray());
    }

    void MoveAgentAlongPath()
    {
        if (currentAgent != null)
        {
            Destroy(currentAgent);
        }

        currentAgent = Instantiate(agentPrefab, waypoints[0], Quaternion.identity);
        NavMeshAgent navMeshAgent = currentAgent.GetComponent<NavMeshAgent>();

        // Calculate the path using NavMesh.CalculatePath
        NavMeshPath navMeshPath = new NavMeshPath();
        NavMesh.CalculatePath(waypoints[0], waypoints[waypoints.Count - 1], NavMesh.AllAreas, navMeshPath);

        // Set the path for the agent
        navMeshAgent.SetPath(navMeshPath);
    }
}
