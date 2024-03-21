using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEngine.UIElements;

public class Line : MonoBehaviour
{
    public LineRenderer lineRenderer;
    List<Vector2> points;

    private Collider objectCollider;

    public void UpdateLine(Vector2 position)
    {
        if (points == null)
        {
            points = new List<Vector2>();
            SetPoint(position);
            return;
        }

        if (Vector2.Distance(points.Last(), position) > .1f)
        {
            SetPoint(position);
        }
    }

    void SetPoint(Vector2 point)
    { 
        points.Add(point);
        lineRenderer.positionCount = points.Count;
        lineRenderer.SetPosition(points.Count - 1, point);
    }

    public void RemovePointFromLineRenderer(Vector2 mousePosition, List<Line> lines, int index)
    {
        List<Vector3> newPositions = new List<Vector3>();

        List<Vector3> leftPositions = new List<Vector3>();
        List<Vector3> rightPositions = new List<Vector3>();

        int intermediatePoint = -1;
        // Copy the positions excluding the point to remove
        for (int i = 0; i < lineRenderer.positionCount; i++)
        {
            Vector3 pointPosition = lineRenderer.GetPosition(i);
            newPositions.Add(pointPosition);

            // Compare the position with the target position using approximate equality
            if (Vector2.Distance(new Vector2(pointPosition.x, pointPosition.y), mousePosition) < 0.3f)
            {
                newPositions.Remove(pointPosition);
                if (i != 0 && i != (lineRenderer.positionCount - 1))
                {
                    // Intermediate Point
                    intermediatePoint = i;
                }
            }
        }

        //if (intermediatePoint != -1)
        //{
        //    for (int i = 0; i < lineRenderer.positionCount; i++)
        //    {
        //        if (i < intermediatePoint)
        //        {
        //            leftPositions.Add(newPositions[i]);
        //        }

        //        if (i > intermediatePoint)
        //        {
        //            rightPositions.Add(newPositions[i]);
        //        }

        //        newPositions = leftPositions;

        //    }
        //}

    
         // Update the LineRenderer with the new positions
         lineRenderer.positionCount = newPositions.Count;
         lineRenderer.SetPositions(newPositions.ToArray());
    
    }

    public bool IsPointInLineBounds(Vector2 point)
    {
        if (lineRenderer == null)
            return false;

        Bounds bounds = lineRenderer.bounds;

        return bounds.Contains(point);
    }



}
