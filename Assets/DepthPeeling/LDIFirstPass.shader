Shader "Unlit/LDIFirstPass"
{
    Properties
    {
        _StencilRef("StencilRef", Integer) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }

        Pass
        {
            
            Cull Off
            ZTest Always
            ZWrite On

            Stencil
            {
                Ref [StencilRef]
                Comp Greater
                ReadMask 255
                WriteMask 255

                Fail IncrSat
                ZFail IncrSat
                Pass IncrSat
            }

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float depth = LinearEyeDepth(i.vertex.z / i.vertex.w);
                return fixed4(depth, depth, depth, 1);

            }
            ENDCG
        }
    }
}
