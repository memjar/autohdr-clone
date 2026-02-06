/**
 * AutoHDR Clone - Type Definitions
 * Database schema types for subscription management
 */

// Subscription tiers
export type SubscriptionTier = 'free' | 'standard' | 'enterprise';
export type SubscriptionStatus = 'active' | 'paused' | 'cancelled' | 'expired';
export type BillingCycle = 'monthly' | 'yearly';

// User account
export interface User {
  id: string;
  email: string;
  name?: string;
  profileImage?: string;
  createdAt: Date;
  updatedAt: Date;

  // Referral
  referralCode: string;
  referralBalance: number;

  // Integrations
  dropboxToken?: string;
  dropboxFolderId?: string;
}

// Subscription
export interface Subscription {
  id: string;
  userId: string;

  tier: SubscriptionTier;
  status: SubscriptionStatus;

  creditsAllowed: number;  // Monthly allowance
  creditsUsed: number;
  creditsRemaining: number;

  priceMonthly: number;
  priceYearly?: number;
  billingCycle: BillingCycle;

  // Stripe
  stripeCustomerId?: string;
  stripeSubscriptionId?: string;

  // Dates
  startDate: Date;
  renewalDate?: Date;
  cancellationDate?: Date;

  // Settings
  autoRenew: boolean;
  creditRollover: boolean;

  createdAt: Date;
  updatedAt: Date;
}

// Style settings (saved per user)
export interface StyleSettings {
  id: string;
  userId: string;

  // Mode
  interiorExterior: 'interior' | 'exterior';

  // Sliders (-2 to 2)
  brightness: number;
  contrast: number;
  vibrance: number;
  whiteBalance: number;

  // Dropdowns
  windowPull: 'natural' | 'strong' | 'subtle';
  cloudStyle: 'fluffy-whispy' | 'dramatic' | 'clear-sky';
  twilightStyle: 'pink' | 'golden' | 'blue' | 'orange';
  tvScreen: 'none' | 'black' | 'nature' | 'abstract';

  // Toggles
  fireplace: boolean;
  declutter: boolean;
  interiorClouds: boolean;
  exteriorClouds: boolean;
  deduplication: boolean;
  walkthrough: boolean;
  retainSky: boolean;
  perspective: boolean;
  grassReplacement: boolean;
  signRemoval: boolean;

  // Output
  filenames: 'default' | 'sequential' | 'original';

  createdAt: Date;
  updatedAt: Date;
}

// Team member
export interface TeamMember {
  id: string;
  userId: string;  // Owner

  email: string;
  role: 'admin' | 'editor' | 'viewer';
  status: 'invited' | 'active' | 'removed';

  invitedAt: Date;
  acceptedAt?: Date;
  removedAt?: Date;
}

// Billing record
export interface BillingRecord {
  id: string;
  subscriptionId: string;

  amount: number;
  currency: string;
  description: string;
  status: 'completed' | 'pending' | 'failed';

  invoiceId?: string;
  invoiceUrl?: string;

  billingDate: Date;
  paidDate?: Date;
}

// Credit log
export interface CreditLog {
  id: string;
  subscriptionId: string;

  action: 'purchase' | 'usage' | 'rollover' | 'adjustment' | 'refund';
  amount: number;  // Positive for add, negative for use
  newBalance: number;

  description?: string;
  photoIds?: string[];  // Photos processed

  createdAt: Date;
}

// Pricing configuration
export const PRICING_CONFIG = {
  free: {
    name: 'Free',
    priceMonthly: 0,
    priceYearly: 0,
    credits: 10,
    features: ['Basic HDR Editing', '10 photos/month']
  },
  standard: {
    name: 'Standard',
    priceMonthly: 265,
    priceYearly: 225.25 * 12,  // 20% off
    credits: 500,
    features: [
      'All HDR Features',
      'All Editing Sliders',
      'Object Removal (AI)',
      'Grass Greening',
      'Credit Rollover',
      'Email Support'
    ]
  },
  enterprise: {
    name: 'Enterprise',
    priceMonthly: 2250,
    priceYearly: 2025 * 12,  // 20% off
    credits: 5000,
    features: [
      'All Standard Features',
      'Auto TV Blackout',
      'Auto Add Fire',
      'Walkthrough Re-ordering',
      'Dedicated Slack Channel',
      'Priority Support 24/7',
      'Dropbox Automation'
    ]
  }
} as const;

// API Response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface SubscriptionResponse {
  subscription: Subscription;
  user: User;
  settings: StyleSettings;
}

export interface ProcessingResponse {
  imageUrl: string;
  processingTimeMs: number;
  creditsUsed: number;
  creditsRemaining: number;
  processorVersion: string;
}
