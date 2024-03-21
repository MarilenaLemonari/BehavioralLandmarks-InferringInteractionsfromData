using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Xml.Linq;

public class JSONLoader : MonoBehaviour
{
    public string folderPath;
    public string fileName;  
    private string filePath;

    class Grid
    {
        //int n_width;
        //int n_height;
        //List<int> gridIndices = new List<int>();

        //Grid(int width, int height)
        //{
        //    this.n_width = width / 5;
        //    this.n_height = height / 5;
        //}

        //void set_gridIndex(int index)
        //{
        //    gridIndices.Add(index);
        //}
        public string index;

    }

    private void Start()
    {
        filePath = Path.Combine(folderPath, fileName);
        if (File.Exists(filePath))
        {
            string jsonText = File.ReadAllText(filePath);
            Debug.Log(jsonText);


        }
        else
        {
            Debug.LogError("JSON file not found: " + filePath);
        }
    }
}
