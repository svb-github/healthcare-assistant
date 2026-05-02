"""
Health calculation tools for the Healthcare Assistant.
These are used by the LangGraph agents as callable tools.

Concepts used from: mission4-reAct-tools (custom tool functions)
"""

import math


def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """
    Calculate Body Mass Index (BMI) and return category.
    
    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
    
    Returns:
        Dictionary with BMI value, category, and health advice
    """
    if weight_kg <= 0 or height_cm <= 0:
        return {"error": "Weight and height must be positive numbers."}
    
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)
    
    if bmi < 18.5:
        category = "Underweight"
        advice = "Consider consulting a nutritionist to achieve a healthy weight through balanced diet."
    elif bmi < 25:
        category = "Normal weight"
        advice = "Great! Maintain your healthy weight with regular exercise and balanced nutrition."
    elif bmi < 30:
        category = "Overweight"
        advice = "Consider lifestyle modifications — regular physical activity and dietary changes can help."
    else:
        category = "Obese"
        advice = "Please consult a healthcare provider for a personalized weight management plan."
    
    return {
        "bmi": bmi,
        "category": category,
        "advice": advice,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
    }


def calculate_daily_calories(weight_kg: float, height_cm: float, age: int, gender: str, activity_level: str) -> dict:
    """
    Calculate daily calorie needs using the Mifflin-St Jeor equation.
    
    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        age: Age in years
        gender: 'male' or 'female'
        activity_level: 'sedentary', 'light', 'moderate', 'active', or 'very_active'
    
    Returns:
        Dictionary with BMR, TDEE, and recommendations
    """
    if weight_kg <= 0 or height_cm <= 0 or age <= 0:
        return {"error": "Weight, height, and age must be positive numbers."}
    
    gender = gender.lower().strip()
    activity_level = activity_level.lower().strip().replace(" ", "_")
    
    # Mifflin-St Jeor Equation
    if gender == "male":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    elif gender == "female":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    else:
        return {"error": "Gender must be 'male' or 'female'."}
    
    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }
    
    multiplier = activity_multipliers.get(activity_level, 1.55)
    tdee = round(bmr * multiplier)
    
    return {
        "bmr": round(bmr),
        "tdee": tdee,
        "weight_loss_calories": tdee - 500,
        "weight_gain_calories": tdee + 500,
        "activity_level": activity_level,
        "gender": gender,
    }


def calculate_heart_rate_zones(age: int, resting_hr: int = 70) -> dict:
    """
    Calculate target heart rate training zones using the Karvonen method.
    
    Args:
        age: Age in years
        resting_hr: Resting heart rate in BPM (default 70)
    
    Returns:
        Dictionary with heart rate zones for different exercise intensities
    """
    if age <= 0 or age > 120:
        return {"error": "Please provide a valid age (1-120)."}
    if resting_hr <= 0 or resting_hr > 220:
        return {"error": "Please provide a valid resting heart rate."}
    
    max_hr = 220 - age
    hr_reserve = max_hr - resting_hr
    
    zones = {
        "max_heart_rate": max_hr,
        "resting_heart_rate": resting_hr,
        "zones": {
            "Zone 1 - Recovery (50-60%)": {
                "low": round(hr_reserve * 0.5 + resting_hr),
                "high": round(hr_reserve * 0.6 + resting_hr),
                "description": "Light activity, warm-up, cool-down",
            },
            "Zone 2 - Fat Burn (60-70%)": {
                "low": round(hr_reserve * 0.6 + resting_hr),
                "high": round(hr_reserve * 0.7 + resting_hr),
                "description": "Endurance training, fat burning",
            },
            "Zone 3 - Cardio (70-80%)": {
                "low": round(hr_reserve * 0.7 + resting_hr),
                "high": round(hr_reserve * 0.8 + resting_hr),
                "description": "Aerobic fitness improvement",
            },
            "Zone 4 - Hard (80-90%)": {
                "low": round(hr_reserve * 0.8 + resting_hr),
                "high": round(hr_reserve * 0.9 + resting_hr),
                "description": "Anaerobic training, speed work",
            },
            "Zone 5 - Max (90-100%)": {
                "low": round(hr_reserve * 0.9 + resting_hr),
                "high": max_hr,
                "description": "Maximum effort, sprint intervals",
            },
        },
    }
    return zones


def calculate_water_intake(weight_kg: float, activity_level: str = "moderate", climate: str = "temperate") -> dict:
    """
    Calculate recommended daily water intake.
    
    Args:
        weight_kg: Weight in kilograms
        activity_level: 'sedentary', 'moderate', or 'active'
        climate: 'cold', 'temperate', or 'hot'
    
    Returns:
        Dictionary with water intake recommendations in liters
    """
    if weight_kg <= 0:
        return {"error": "Weight must be a positive number."}
    
    # Base: 35ml per kg body weight
    base_ml = weight_kg * 35
    
    activity_multipliers = {"sedentary": 1.0, "moderate": 1.2, "active": 1.5}
    climate_multipliers = {"cold": 0.9, "temperate": 1.0, "hot": 1.3}
    
    activity_mult = activity_multipliers.get(activity_level.lower().strip(), 1.2)
    climate_mult = climate_multipliers.get(climate.lower().strip(), 1.0)
    
    total_ml = base_ml * activity_mult * climate_mult
    total_liters = round(total_ml / 1000, 1)
    glasses = math.ceil(total_ml / 250)  # 250ml per glass
    
    return {
        "daily_liters": total_liters,
        "daily_glasses": glasses,
        "daily_ml": round(total_ml),
        "tip": f"Drink about {glasses} glasses (250ml each) spread throughout the day.",
    }


def assess_symptom_severity(symptoms: list[str], duration_days: int, has_fever: bool = False, age: int = 30) -> dict:
    """
    Provide a basic symptom severity assessment. NOT a medical diagnosis.
    
    Args:
        symptoms: List of symptom strings
        duration_days: How many days symptoms have persisted
        has_fever: Whether the patient has fever
        age: Patient age
    
    Returns:
        Dictionary with severity level and recommendation
    """
    if not symptoms:
        return {"error": "Please provide at least one symptom."}
    
    # Emergency symptoms that always warrant immediate medical attention
    emergency_keywords = [
        "chest pain", "difficulty breathing", "shortness of breath",
        "severe bleeding", "unconscious", "seizure", "stroke",
        "sudden numbness", "severe headache", "vision loss",
        "suicidal", "self harm", "poisoning", "allergic reaction",
        "swelling of face", "swelling of throat",
    ]
    
    # Check for emergency symptoms
    symptoms_lower = [s.lower().strip() for s in symptoms]
    for symptom in symptoms_lower:
        for emergency in emergency_keywords:
            if emergency in symptom:
                return {
                    "severity": "EMERGENCY",
                    "recommendation": "🚨 SEEK IMMEDIATE MEDICAL ATTENTION. Call emergency services or go to the nearest emergency room NOW.",
                    "symptoms_reported": symptoms,
                    "disclaimer": "This is not a medical diagnosis. Always consult a qualified healthcare professional.",
                }
    
    # Severity scoring
    score = 0
    score += len(symptoms) * 2  # More symptoms = higher severity
    score += min(duration_days, 14)  # Duration adds to severity (capped at 14)
    if has_fever:
        score += 5
    if age > 65 or age < 5:
        score += 5  # Vulnerable age groups
    
    if score <= 5:
        severity = "MILD"
        recommendation = "🟢 Your symptoms appear mild. Rest, stay hydrated, and monitor. If symptoms worsen or persist beyond 3 days, consult a doctor."
    elif score <= 15:
        severity = "MODERATE"
        recommendation = "🟡 Your symptoms suggest you should consult a doctor soon — ideally within 24-48 hours. Rest and monitor for any worsening."
    else:
        severity = "HIGH"
        recommendation = "🔴 Your symptoms are concerning. Please consult a healthcare professional TODAY. Do not delay seeking medical advice."
    
    return {
        "severity": severity,
        "score": score,
        "recommendation": recommendation,
        "symptoms_reported": symptoms,
        "duration_days": duration_days,
        "disclaimer": "⚕️ DISCLAIMER: This is NOT a medical diagnosis. It is a preliminary assessment only. Always consult a qualified healthcare professional for proper diagnosis and treatment.",
    }
