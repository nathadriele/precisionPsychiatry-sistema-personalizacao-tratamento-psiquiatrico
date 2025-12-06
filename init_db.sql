-- Inicialização do banco de dados PostgreSQL para Medicina de Precisão em Psiquiatria

-- ============================================================
-- Tabelas Principais
-- ============================================================

-- Pacientes
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) UNIQUE NOT NULL,
    age INTEGER NOT NULL,
    gender CHAR(1) NOT NULL,
    bmi FLOAT,
    education_years INTEGER,
    employment_status VARCHAR(50),
    ethnicity VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patients_patient_id ON patients(patient_id);
CREATE INDEX idx_patients_created_at ON patients(created_at);

-- ============================================================
-- Dados Genômicos
-- ============================================================

CREATE TABLE IF NOT EXISTS genomic_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    
    -- Genótipos
    cyp2d6_genotype INTEGER,
    cyp3a4_genotype INTEGER,
    cyp1a2_genotype INTEGER,
    cyp2c19_genotype INTEGER,
    cyp2b6_genotype INTEGER,
    mthfr_genotype INTEGER,
    comt_genotype INTEGER,
    bdnf_genotype INTEGER,
    htr1a_genotype INTEGER,
    tph1_genotype INTEGER,
    slc6a4_genotype INTEGER,
    
    -- Status de metabolizador
    metabolizer_status VARCHAR(20),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_genomic_patient_id ON genomic_data(patient_id);

-- ============================================================
-- Biomarcadores Neurobiológicos
-- ============================================================

CREATE TABLE IF NOT EXISTS neurobiological_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    
    -- Inflamação
    il6_pg_ml FLOAT,
    tnf_alpha_pg_ml FLOAT,
    crp_mg_l FLOAT,
    
    -- Eixo HPA
    cortisol_morning_nmol_l FLOAT,
    cortisol_evening_nmol_l FLOAT,
    acth_pm_mlU_l FLOAT,
    
    -- Neurotransmissores
    serotonin_ng_ml FLOAT,
    dopamine_pg_ml FLOAT,
    noradrenaline_pg_ml FLOAT,
    
    -- BDNF
    bdnf_ng_ml FLOAT,
    
    -- Triptofano/Quinurenina
    kynurenine_nmol_l FLOAT,
    tryptophan_nmol_l FLOAT,
    
    collection_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_neurobiological_patient_id ON neurobiological_data(patient_id);
CREATE INDEX idx_neurobiological_collection_date ON neurobiological_data(collection_date);

-- ============================================================
-- Dados Psicossociais
-- ============================================================

CREATE TABLE IF NOT EXISTS psychosocial_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    
    -- Escalas psicométricas
    phq9_score INTEGER,
    gad7_score INTEGER,
    panss_positive INTEGER,
    panss_negative INTEGER,
    ctq_total INTEGER,
    ace_total INTEGER,
    psqi_score INTEGER,
    
    -- Suporte social
    social_support_score FLOAT,
    life_stressors INTEGER,
    
    assessment_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_psychosocial_patient_id ON psychosocial_data(patient_id);
CREATE INDEX idx_psychosocial_assessment_date ON psychosocial_data(assessment_date);

-- ============================================================
-- Medicamentos (Histórico)
-- ============================================================

CREATE TABLE IF NOT EXISTS medication_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    
    medication_name VARCHAR(100) NOT NULL,
    dosage VARCHAR(50),
    start_date DATE,
    end_date DATE,
    reason_for_discontinuation VARCHAR(255),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_medication_history_patient_id ON medication_history(patient_id);

-- ============================================================
-- Resultados de Tratamento
-- ============================================================

CREATE TABLE IF NOT EXISTS treatment_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    
    baseline_phq9 INTEGER,
    baseline_date DATE,
    
    week_4_phq9 INTEGER,
    week_8_phq9 INTEGER,
    week_12_phq9 INTEGER,
    
    response_status VARCHAR(50),  -- responder, non-responder, partial_responder, refractory
    is_responder BOOLEAN,
    is_refractory BOOLEAN,
    
    treatment_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_treatment_outcomes_patient_id ON treatment_outcomes(patient_id);
CREATE INDEX idx_treatment_outcomes_response_status ON treatment_outcomes(response_status);

-- ============================================================
-- Predições do Modelo
-- ============================================================

CREATE TABLE IF NOT EXISTS model_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    
    model_version VARCHAR(50),
    model_type VARCHAR(50),
    
    prediction_probability FLOAT NOT NULL,
    risk_category VARCHAR(50),
    predicted_response BOOLEAN,
    confidence FLOAT,
    
    input_features JSONB,
    explanation JSONB,
    
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_predictions_patient_id ON model_predictions(patient_id);
CREATE INDEX idx_model_predictions_prediction_date ON model_predictions(prediction_date);
CREATE INDEX idx_model_predictions_model_version ON model_predictions(model_version);

-- ============================================================
-- Recomendações de Medicamentos
-- ============================================================

CREATE TABLE IF NOT EXISTS medication_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    
    medication_name VARCHAR(100) NOT NULL,
    reason VARCHAR(255),
    efficacy_score FLOAT,
    caution_flag BOOLEAN DEFAULT FALSE,
    contraindication_flag BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_medication_recommendations_patient_id ON medication_recommendations(patient_id);

-- ============================================================
-- Logs de Auditoria
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE SET NULL,
    
    action VARCHAR(100) NOT NULL,
    action_type VARCHAR(50),  -- create, read, update, delete, predict
    entity_type VARCHAR(50),
    changes JSONB,
    
    user_ip VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_patient_id ON audit_logs(patient_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_logs_action_type ON audit_logs(action_type);

-- ============================================================
-- Views Úteis
-- ============================================================

-- View: Dados consolidados de paciente
CREATE OR REPLACE VIEW v_patient_complete AS
SELECT
    p.id,
    p.patient_id,
    p.age,
    p.gender,
    p.bmi,
    p.education_years,
    gd.metabolizer_status,
    gd.cyp2d6_genotype,
    nb.il6_pg_ml,
    nb.cortisol_morning_nmol_l,
    nb.bdnf_ng_ml,
    ps.phq9_score,
    ps.gad7_score,
    ps.ctq_total,
    to.is_responder,
    to.response_status,
    p.created_at
FROM patients p
LEFT JOIN genomic_data gd ON p.id = gd.patient_id
LEFT JOIN neurobiological_data nb ON p.id = nb.patient_id
LEFT JOIN psychosocial_data ps ON p.id = ps.patient_id
LEFT JOIN treatment_outcomes to ON p.id = to.patient_id;

-- View: Pacientes em risco de não-resposta
CREATE OR REPLACE VIEW v_high_risk_patients AS
SELECT
    p.patient_id,
    p.age,
    ps.phq9_score,
    ps.ctq_total,
    nb.il6_pg_ml,
    mp.risk_category,
    mp.prediction_probability
FROM patients p
JOIN psychosocial_data ps ON p.id = ps.patient_id
JOIN neurobiological_data nb ON p.id = nb.patient_id
LEFT JOIN model_predictions mp ON p.id = mp.patient_id
WHERE 
    (ps.phq9_score >= 20 OR ps.ctq_total >= 100 OR nb.il6_pg_ml > 5)
    OR mp.risk_category = 'high';

-- ============================================================
-- Funções Úteis
-- ============================================================

-- Função para atualizar updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para atualizar updated_at em patients
CREATE TRIGGER patients_updated_at BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger para atualizar updated_at em treatment_outcomes
CREATE TRIGGER treatment_outcomes_updated_at BEFORE UPDATE ON treatment_outcomes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Dados Iniciais
-- ============================================================

-- Inserir um paciente de exemplo
INSERT INTO patients (patient_id, age, gender, bmi, education_years, employment_status, ethnicity)
VALUES ('PAT_00000', 45, 'F', 24.5, 16, 'employed', 'European')
ON CONFLICT (patient_id) DO NOTHING;

-- ============================================================
-- Permissões
-- ============================================================

-- Conceder permissões ao usuário da aplicação
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

COMMIT;