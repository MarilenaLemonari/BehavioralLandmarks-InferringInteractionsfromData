using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class CSVReader : MonoBehaviour
{
    public string csvFilePath = "Assets\\SavedDots\\.APpark__dots_0.csv"; // Adjust the path to your CSV file

    public List<Vector3> waypoints = new List<Vector3>();
    public float offset_x;
    public float offset_z;

    void Awake()
    {
        ReadCSVFile();
        // Use the 'waypoints' list as needed in your project
        Debug.Log("Number of waypoints: " + waypoints.Count);
    }

    void ReadCSVFile()
    {
        try
        {
            int counter = 0;
            float firstX = 0;
            float firstZ = 0;
            using (StreamReader reader = new StreamReader(csvFilePath))
            {
                while (!reader.EndOfStream)
                {
                    string line = reader.ReadLine();
                    string[] values = line.Split(',');

                    counter++;
                    if (counter == 1)
                    {
                        continue;
                    }

                    if (values.Length >= 2)
                    {
                        float x = float.Parse(values[0]);
                        float y = float.Parse(values[1]);

                        if (firstX == 0 && x > 0)
                        {
                            firstX = x + 10;
                            firstZ = y;
                        }

                        waypoints.Add(new Vector3(x - firstX + offset_x, 0f, y - firstZ +offset_z) * 3f);
                    }
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError("Error reading CSV file: " + e.Message);
        }

        int i = 0;
        foreach (Transform child in transform)
        {
            //child.position = waypoints[0] + new Vector3(i, 0f, i);
            //child.gameObject.SetActive(true);
            //i++;

            if (i == 0)
            {
                child.position = waypoints[0];
                child.gameObject.SetActive(true);
            }

            if (i == 1)
            {
                child.position = waypoints[0] + new Vector3(1.1f, 0f, 1.0f);
                child.gameObject.SetActive(true);
            }

            if (i == 2)
            {
                child.position = waypoints[0] + new Vector3(1.2f, 0f, -1.2f);
                child.gameObject.SetActive(true);
            }

            if (i == 3)
            {
                child.position = waypoints[0] + new Vector3(-1.3f, 0f, 1);
                child.gameObject.SetActive(true);
            }

            if (i == 4)
            {
                child.position = waypoints[0] + new Vector3(-1.3f, 0f, -1.1f);
                child.gameObject.SetActive(true);
            }

            i++;

        }
    }

    private void OnDrawGizmos()
    {
        if(waypoints.Count > 0)
        {
            foreach (var waypoint in waypoints)
            {
                Gizmos.DrawCube(waypoint, new Vector3(0.3f, 0.3f, 0.3f));
            }
        }
    }
}
