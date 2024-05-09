using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class LDI : MonoBehaviour
{
    public Vector2Int ldi_resolution;
    public List<RenderTexture> ldi_hierarchy = new List<RenderTexture>();
    public Material material_first_pass;
    public Material material_second_pass;
    public ComputeShader compute_ldi;
    public Camera camera;
    public uint max_ldi_hierarchy_size;
    public RenderTexture depth_complexities;
    public BoxCollider AABB;

    CommandBuffer command_buffer;
    MeshFilter mesh_filter;
    
    ComputeBuffer does_collide_buffer;

    static Vector3 default_camera_position = new Vector3(0, 0, 10);
    static Vector3 default_camera_target = new Vector3(0, 0, 0);
    static Vector3 default_camera_viewbox_size = new Vector3(16, 16, 16);

    Dictionary<GameObject, LDI> collisions = new Dictionary<GameObject, LDI>();

    bool is_initialized = false;
    void init()
    {
        if(is_initialized) return;

        if (gameObject.TryGetComponent<MeshFilter>(out mesh_filter))
        {
            for (int i = 0; i < max_ldi_hierarchy_size; i++)
            {
                ldi_hierarchy.Add(new RenderTexture(ldi_resolution.x, ldi_resolution.y,
                    UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_SNorm,
                    UnityEngine.Experimental.Rendering.GraphicsFormat.D24_UNorm_S8_UInt,
                    0));
                ldi_hierarchy[i].depthStencilFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.D24_UNorm_S8_UInt;
                ldi_hierarchy[i].stencilFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R8_UInt;
                ldi_hierarchy[i].enableRandomWrite = true;
            }

            command_buffer = new CommandBuffer();

            depth_complexities = new RenderTexture(ldi_resolution.x, ldi_resolution.y, 0, RenderTextureFormat.R8, 0);
            depth_complexities.enableRandomWrite = true;

            does_collide_buffer = new ComputeBuffer(1, sizeof(int));

            AABB = gameObject.AddComponent<BoxCollider>();
            AABB.size = mesh_filter.mesh.bounds.size;
        }

        is_initialized = true;
    }

    void render_ldi_layer(Material material, int ldi_hierarchy_target_index)
    {
        command_buffer.Clear();
        command_buffer.SetRenderTarget(ldi_hierarchy[ldi_hierarchy_target_index], 0);
        command_buffer.SetViewProjectionMatrices(camera.worldToCameraMatrix, camera.projectionMatrix);

        command_buffer.ClearRenderTarget(RTClearFlags.All, Color.clear, 1.0f, 0);
        command_buffer.DrawMesh(mesh_filter.sharedMesh, transform.localToWorldMatrix, material, 0, 0);
        Graphics.ExecuteCommandBuffer(command_buffer);
        command_buffer.Clear();
    }

    void render_ldi_hierarchy()
    {
        material_first_pass.SetInteger("StencilRef", 1);
        render_ldi_layer(material_first_pass, 0);
        for (int i = 1; i < max_ldi_hierarchy_size; i++)
        {
            material_second_pass.SetInteger("StencilRef", i + 1);
            render_ldi_layer(material_second_pass, i);
        }
    }

    // sort ldi_hierarchy[index] and ldi_hierarchy[index+1]
    void _sort_ldi_hierarchy_pass(int comparison_index)
    {
        int kernel = compute_ldi.FindKernel("sort_textures");
        compute_ldi.SetTexture(kernel, "texture_a", ldi_hierarchy[comparison_index]);
        compute_ldi.SetTexture(kernel, "texture_b", ldi_hierarchy[comparison_index + 1]);
        compute_ldi.SetVector("texture_resolution", new Vector2(ldi_hierarchy[comparison_index].width, ldi_hierarchy[comparison_index].height));

        compute_ldi.Dispatch(kernel, Mathf.CeilToInt(ldi_hierarchy[comparison_index].width / 8.0f), Mathf.CeilToInt(ldi_hierarchy[comparison_index].height / 8.0f), 1);
    }

    void sort_ldi_hierarchy()
    {
        for (int iteration = 0; iteration < ldi_hierarchy.Count; iteration++)
        {
            for (int i = 0; i < ldi_hierarchy.Count - 1; i++)
            {
                _sort_ldi_hierarchy_pass(i);
            }
        }
    }

    void blit_stencil_to_color_texture(RenderTexture source_stencil, RenderTexture target_color)
    {
        int kernel = compute_ldi.FindKernel("blit_stencil_to_color");
        compute_ldi.SetTexture(kernel, "stencil_texture", source_stencil, 0, RenderTextureSubElement.Stencil);
        compute_ldi.SetTexture(kernel, "target_color_texture", target_color);
        compute_ldi.SetVector("texture_resolution", new Vector2(source_stencil.width, source_stencil.height));
        compute_ldi.SetFloat("max_stencil_value", (float)ldi_hierarchy.Count);

        compute_ldi.Dispatch(kernel, Mathf.CeilToInt(source_stencil.width / 8.0f), Mathf.CeilToInt(source_stencil.height / 8.0f), 1);
    }


    // sort ldi_hierarchy[index] and ldi_hierarchy[index+1]
    void _sort_ldi_hierarchy_with_depth_complexity_pass(RenderTexture depth_complexities, int hierarchy_size, int comparison_index)
    {
        int kernel = compute_ldi.FindKernel("sort_textures_with_depth_complexities");
        compute_ldi.SetTexture(kernel, "texture_a", ldi_hierarchy[comparison_index]);
        compute_ldi.SetTexture(kernel, "texture_b", ldi_hierarchy[comparison_index + 1]);
        compute_ldi.SetVector("texture_resolution", new Vector2(ldi_hierarchy[comparison_index].width, ldi_hierarchy[comparison_index].height));

        compute_ldi.SetTexture(kernel, "depth_complexities", depth_complexities);
        compute_ldi.SetFloat("depth_complexities_max_value", hierarchy_size);
        compute_ldi.SetInt("comperison_index", comparison_index);

        compute_ldi.Dispatch(kernel, Mathf.CeilToInt(ldi_hierarchy[comparison_index].width / 8.0f), Mathf.CeilToInt(ldi_hierarchy[comparison_index].height / 8.0f), 1);
    }

    void sort_ldi_hierarchy_with_depth_complexity(RenderTexture depth_complexities)
    {
        for (int iteration = 0; iteration < ldi_hierarchy.Count; iteration++)
        {
            for (int i = 0; i < ldi_hierarchy.Count - 1; i++)
            {
                _sort_ldi_hierarchy_with_depth_complexity_pass(depth_complexities, ldi_hierarchy.Count, i);
            }
        }
    }

    void _does_collide_step(RenderTexture a_begin, RenderTexture a_end, Matrix4x4 screen_to_world_matrix_a, RenderTexture b_begin, RenderTexture b_end, Matrix4x4 screen_to_world_matrix_b)
    {
        int kernel = compute_ldi.FindKernel("does_collide");
        compute_ldi.SetTexture(kernel, "object_a_begin", a_begin);
        compute_ldi.SetTexture(kernel, "object_a_end", a_end);
        compute_ldi.SetTexture(kernel, "object_b_begin", b_begin);
        compute_ldi.SetTexture(kernel, "object_b_end", b_end);
        compute_ldi.SetBuffer(kernel, "does_collide_buffer", does_collide_buffer);
        compute_ldi.SetVector("texture_resolution", new Vector2(a_begin.width, a_begin.height));
        compute_ldi.SetMatrix("screen_to_world_matrix_a", screen_to_world_matrix_a);
        compute_ldi.SetMatrix("screen_to_world_matrix_b", screen_to_world_matrix_b);

        compute_ldi.Dispatch(kernel, Mathf.CeilToInt(a_begin.width / 8.0f), Mathf.CeilToInt(a_begin.height / 8.0f), 1);
    }

    void _set_collide_buffer()
    {
        int kernel = compute_ldi.FindKernel("set_collide_buffer");
        compute_ldi.SetBuffer(kernel, "does_collide_buffer", does_collide_buffer);

        compute_ldi.Dispatch(kernel, 1, 1, 1);
    }

    void _reset_collide_buffer()
    {
        int kernel = compute_ldi.FindKernel("reset_collide_buffer");
        compute_ldi.SetBuffer(kernel, "does_collide_buffer", does_collide_buffer);

        compute_ldi.Dispatch(kernel, 1, 1, 1);
    }

    void compute_collision(List<RenderTexture> ldi0, Matrix4x4 screen_to_world_matrix0, List<RenderTexture> ldi1, Matrix4x4 screen_to_world_matrix1)
    {
        _reset_collide_buffer();

        int n1 = ldi0.Count;
        int n2 = ldi1.Count;

        for (int i = 0; i < n1; i += 2)
        {
            for (int j = 0; j < n2; j += 2)
            {
                _does_collide_step(ldi0[i], ldi0[i+1], screen_to_world_matrix0, ldi1[j], ldi1[j+1], screen_to_world_matrix1);
            }
        }

        int[] collision_result = new int[1];
        does_collide_buffer.GetData(collision_result, 0, 0, 1);
        Debug.Log(collision_result[0]);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collisions.ContainsKey(collision.gameObject)) return;
        
        LDI ldi = collision.gameObject.GetComponent<LDI>();
        collisions.Add(collision.gameObject, ldi);
    }

    private void OnCollisionExit(Collision collision)
    {
        collisions.Remove(collision.gameObject);   
    }

    private void Start()
    {
        init();

        render_ldi_hierarchy();
        blit_stencil_to_color_texture(ldi_hierarchy[0], depth_complexities);

        sort_ldi_hierarchy_with_depth_complexity(depth_complexities);

    }

    static float collision_test_period_miliseconds = 16;
    static float time_since_collision_test_miliseconds = 0;

    void Update()
    {
        if (mesh_filter == null)
            return;

        time_since_collision_test_miliseconds += Time.deltaTime;
        if (time_since_collision_test_miliseconds >= collision_test_period_miliseconds)
        {
            
            compute_collision(ldi_hierarchy, Matrix4x4.identity, ldi_hierarchy, Matrix4x4.identity);


            time_since_collision_test_miliseconds -= collision_test_period_miliseconds;
        }
    }


}
