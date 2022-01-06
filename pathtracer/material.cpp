#include "material.h"
#include "sampling.h"

namespace pathtracer
{
///////////////////////////////////////////////////////////////////////////
// A Lambertian (diffuse) material
///////////////////////////////////////////////////////////////////////////
vec3 Diffuse::f(const vec3& wi, const vec3& wo, const vec3& n)
{
	if(dot(wi, n) <= 0.0f)
		return vec3(0.0f);
	if(!sameHemisphere(wi, wo, n))
		return vec3(0.0f);
	return (1.0f / M_PI) * color;
}

vec3 Diffuse::sample_wi(vec3& wi, const vec3& wo, const vec3& n, float& p)
{
	vec3 tangent = normalize(perpendicular(n));
	vec3 bitangent = normalize(cross(tangent, n));
	vec3 sample = cosineSampleHemisphere();
	wi = normalize(sample.x * tangent + sample.y * bitangent + sample.z * n);
	if(dot(wi, n) <= 0.0f)
		p = 0.0f;
	else
		p = max(0.0f, dot(n, wi)) / M_PI;
	return f(wi, wo, n);
}

///////////////////////////////////////////////////////////////////////////
// A Blinn Phong Dielectric Microfacet BRFD
///////////////////////////////////////////////////////////////////////////
vec3 BlinnPhong::refraction_brdf(const vec3& wi, const vec3& wo, const vec3& n)
{
	vec3 wh = normalize(wi + wo);
	float F = (R0 + (1.0 - R0) * pow((1 - abs(dot(wh, wi))), 5));
	if (!refraction_layer) {
		return vec3(0);
	}
	else {
		return (1.0f - F) * refraction_layer->f(wi, wo, n);
	}
}
vec3 BlinnPhong::reflection_brdf(const vec3& wi, const vec3& wo, const vec3& n)
{
	if (dot(n, wi) < 0 || dot(n, wo) < 0)
	{
		return vec3(0.0f);
	}
	vec3 wh = normalize(wi + wo);
	float F = R0 + (1.0 - R0) * pow((1 - abs(dot(wh, wi))), 5);
	float s = shininess;
	float D = (s + 2) / (2 * M_PI) * pow(dot(n, wh), s);
	float G = min(1.0f, min(2 * dot(n, wh) * dot(n, wo) / dot(wo, wh), 2 * dot(n, wh) * dot(n, wi) / dot(wo, wh)));
	float brdf = F * D * G / (4 * dot(n, wo) * dot(n, wi));
	return vec3(brdf);
}

vec3 BlinnPhong::f(const vec3& wi, const vec3& wo, const vec3& n)
{
	return reflection_brdf(wi, wo, n) + refraction_brdf(wi, wo, n);
}

vec3 BlinnPhong::sample_wi(vec3& wi, const vec3& wo, const vec3& n, float& p)
{
	vec3 tangent = normalize(perpendicular(n));
	vec3 bitangent = normalize(cross(tangent, n));
	float phi = 2.0f * M_PI * randf();
	float cos_theta = pow(randf(), 1.0f / (shininess + 1));
	float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
	vec3 wh = normalize(sin_theta * cos(phi) * tangent + sin_theta * sin(phi) * bitangent + cos_theta * n);
	if (dot(wo, n) < 0.0f) return vec3(0.0f);
	wi = normalize(2 * dot(wh, wo) * wh - wo);
	float pwh = (shininess + 1) * pow(dot(n, wh), shininess) / (2 * M_PI);
	p = pwh / (4 * dot(wo, wh));
	p = p * 0.5f;
	if (randf() < 0.5) {
		return reflection_brdf(wi, wo, n);
	}
	else {
		if (!refraction_layer) {
			return vec3(0);
		}
		else {
			vec3 brdf = refraction_layer->sample_wi(wi, wo, n, p);
			float f = R0 + (1.0f - R0) * pow(1.0f - abs(dot(wh, wi)), 5.0f);
			return (1 - f) * brdf;
		}
	}
}

///////////////////////////////////////////////////////////////////////////
// A Blinn Phong Metal Microfacet BRFD (extends the BlinnPhong class)
///////////////////////////////////////////////////////////////////////////
vec3 BlinnPhongMetal::refraction_brdf(const vec3& wi, const vec3& wo, const vec3& n)
{
	return vec3(0.0f);
}
vec3 BlinnPhongMetal::reflection_brdf(const vec3& wi, const vec3& wo, const vec3& n)
{
	return BlinnPhong::reflection_brdf(wi, wo, n) * color;
};

///////////////////////////////////////////////////////////////////////////
// A Linear Blend between two BRDFs
///////////////////////////////////////////////////////////////////////////
vec3 LinearBlend::f(const vec3& wi, const vec3& wo, const vec3& n)
{
	return w * bsdf0->f(wi, wo, n) + (1 - w) * bsdf1->f(wi, wo, n);
}

vec3 LinearBlend::sample_wi(vec3& wi, const vec3& wo, const vec3& n, float& p)
{
	//p = 0.0f;
	if (randf() < w) {
		return bsdf0->sample_wi(wi, wo, n, p);
	}
	else {
		return bsdf1->sample_wi(wi, wo, n, p);
	}
}

vec3 Glass::f(const vec3& wi, const vec3& wo, const vec3& n)
{
	return vec3(0.0);
}

vec3 refractionD(vec3 wo, vec3 normal, float index1, float index2) {
	float n12 = index1 / index2;
	float cos_won = dot(wo, normal);
	vec3 refraD1 = float(-sqrt(1 - pow(n12, 2) * (1 - pow(cos_won, 2)))) * normal;
	vec3 refraD2 = n12 * (-wo + abs(cos_won) * normal);
	return normalize(refraD1 + refraD2);
}

vec3 Glass::sample_wi(vec3& wi, const vec3& wo, const vec3& n, float& p)
{
	float flag = dot(wo, n);
	if (flag < -0.0001) {
		// from inside to outside
		vec3 shading_normal = -n;
		float eta = refract_factor;

		float total_reflection = pow(refract_factor, 2) * (1 - pow(dot(wo, shading_normal), 2));
		if (total_reflection > 1) {  // total reflection
			wi = normalize(2 * dot(shading_normal, wo) * shading_normal - wo);
			p = abs(dot(wi, shading_normal));
			return vec3(1.0f);
		}
		wi = refractionD(wo, shading_normal, 1.5, 1);
		p = abs(dot(wi, shading_normal));
		return vec3(1.0f);
	}
	else {
		// from outside to inside
		vec3 shading_normal = n;
		wi = refractionD(wo, shading_normal, 1, 1.5);
		p = abs(dot(wi, shading_normal));
		return vec3(1.0f);
	}
}

vec3 Glass::refraction_brdf(const vec3& wi, const vec3& wo, const vec3& n)
{
	return vec3(0);
}

vec3 Glass::reflection_brdf(const vec3& wi, const vec3& wo, const vec3& n)
{
	if (dot(n, wi) < 0 || dot(n, wo) < 0)
	{
		return vec3(0.0f);
	}
	vec3 wh = normalize(wi + wo);
	float F = R0 + (1.0 - R0) * pow((1 - abs(dot(wh, wi))), 5);
	float s = shininess;
	float D = (s + 2) / (2 * M_PI) * pow(dot(n, wh), s);
	float G = min(1.0f, min(2 * dot(n, wh) * dot(n, wo) / dot(wo, wh), 2 * dot(n, wh) * dot(n, wi) / dot(wo, wh)));
	float brdf = F * D * G / (4 * dot(n, wo) * dot(n, wi));
	return vec3(brdf);
}

vec3 Glass::btdf(const vec3& wi, const vec3& wo, const vec3& n, float n1, float n2)
{
	if (dot(n, wi) < 0 || dot(n, wo) < 0)
	{
		return vec3(0.0f);
	}
	vec3 ht = -(n1 * wi + n2 * wo); // half vector of refraction
	float F = R0 + (1.0 - R0) * pow((1 - abs(dot(ht, wi))), 5);
	float s = shininess;
	float D = (s + 2) / (2 * M_PI) * pow(dot(n, ht), s);
	float G = min(1.0f, min(2 * dot(n, ht) * dot(n, wo) / dot(wo, ht), 2 * dot(n, ht) * dot(n, wi) / dot(wo, ht)));
	float d_iht = dot(wi, ht);
	float d_oht = dot(wo, ht);
	float btdf_1 = abs(d_iht * d_oht / (dot(wi, n) * dot(wo, n)));
	float btdf_2 = pow(n2, 2) * (1.0 - F) * G * D / pow(n1 * d_iht + n2 * d_oht, 2);
	float btdf = btdf_1 * btdf_2;
	return vec3(btdf);
}

///////////////////////////////////////////////////////////////////////////
// A perfect specular refraction.
///////////////////////////////////////////////////////////////////////////
} // namespace pathtracer