/**
 * Subscription API Routes
 * Handles subscription management, upgrades, and credit purchases
 */

import { NextRequest, NextResponse } from 'next/server';
import { PRICING_CONFIG, Subscription, SubscriptionTier } from '@/lib/types';

// In-memory store for demo (replace with database in production)
const subscriptions = new Map<string, Subscription>();

// Initialize demo subscription
const DEMO_USER_ID = 'demo-user';
subscriptions.set(DEMO_USER_ID, {
  id: 'sub_demo',
  userId: DEMO_USER_ID,
  tier: 'free',
  status: 'active',
  creditsAllowed: 10,
  creditsUsed: 0,
  creditsRemaining: 10,
  priceMonthly: 0,
  billingCycle: 'monthly',
  startDate: new Date(),
  autoRenew: false,
  creditRollover: false,
  createdAt: new Date(),
  updatedAt: new Date(),
});

// GET /api/subscription - Get current subscription
export async function GET(req: NextRequest) {
  const userId = req.headers.get('x-user-id') || DEMO_USER_ID;

  const subscription = subscriptions.get(userId);

  if (!subscription) {
    return NextResponse.json(
      { success: false, error: 'No subscription found' },
      { status: 404 }
    );
  }

  return NextResponse.json({
    success: true,
    data: subscription,
  });
}

// POST /api/subscription - Create or upgrade subscription
export async function POST(req: NextRequest) {
  const userId = req.headers.get('x-user-id') || DEMO_USER_ID;

  try {
    const body = await req.json();
    const { tier, billingCycle } = body as {
      tier: SubscriptionTier;
      billingCycle: 'monthly' | 'yearly';
    };

    if (!tier || !billingCycle) {
      return NextResponse.json(
        { success: false, error: 'Missing tier or billingCycle' },
        { status: 400 }
      );
    }

    const pricing = PRICING_CONFIG[tier];
    if (!pricing) {
      return NextResponse.json(
        { success: false, error: 'Invalid tier' },
        { status: 400 }
      );
    }

    const price = billingCycle === 'monthly' ? pricing.priceMonthly : pricing.priceYearly / 12;

    const existingSubscription = subscriptions.get(userId);

    // Calculate credits (rollover if upgrading)
    let creditsRemaining = pricing.credits;
    if (existingSubscription && existingSubscription.creditRollover) {
      creditsRemaining += existingSubscription.creditsRemaining;
    }

    const newSubscription: Subscription = {
      id: `sub_${Date.now()}`,
      userId,
      tier,
      status: 'active',
      creditsAllowed: pricing.credits,
      creditsUsed: 0,
      creditsRemaining,
      priceMonthly: price,
      priceYearly: pricing.priceYearly,
      billingCycle,
      startDate: new Date(),
      renewalDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
      autoRenew: true,
      creditRollover: tier !== 'free',
      createdAt: existingSubscription?.createdAt || new Date(),
      updatedAt: new Date(),
    };

    subscriptions.set(userId, newSubscription);

    return NextResponse.json({
      success: true,
      data: newSubscription,
      message: `Upgraded to ${tier} plan with ${creditsRemaining} credits`,
    });
  } catch (error) {
    console.error('Subscription error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to update subscription' },
      { status: 500 }
    );
  }
}

// PUT /api/subscription - Use credits
export async function PUT(req: NextRequest) {
  const userId = req.headers.get('x-user-id') || DEMO_USER_ID;

  try {
    const body = await req.json();
    const { photosProcessed } = body as { photosProcessed: number };

    if (!photosProcessed || photosProcessed < 1) {
      return NextResponse.json(
        { success: false, error: 'Invalid photosProcessed count' },
        { status: 400 }
      );
    }

    const subscription = subscriptions.get(userId);

    if (!subscription) {
      return NextResponse.json(
        { success: false, error: 'No subscription found' },
        { status: 404 }
      );
    }

    if (subscription.creditsRemaining < photosProcessed) {
      return NextResponse.json(
        {
          success: false,
          error: 'Insufficient credits',
          creditsRemaining: subscription.creditsRemaining,
          creditsNeeded: photosProcessed,
        },
        { status: 402 } // Payment Required
      );
    }

    // Deduct credits
    subscription.creditsUsed += photosProcessed;
    subscription.creditsRemaining -= photosProcessed;
    subscription.updatedAt = new Date();

    subscriptions.set(userId, subscription);

    return NextResponse.json({
      success: true,
      data: {
        creditsUsed: photosProcessed,
        creditsRemaining: subscription.creditsRemaining,
        totalUsedThisMonth: subscription.creditsUsed,
      },
    });
  } catch (error) {
    console.error('Credit usage error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to use credits' },
      { status: 500 }
    );
  }
}

// DELETE /api/subscription - Cancel subscription
export async function DELETE(req: NextRequest) {
  const userId = req.headers.get('x-user-id') || DEMO_USER_ID;

  const subscription = subscriptions.get(userId);

  if (!subscription) {
    return NextResponse.json(
      { success: false, error: 'No subscription found' },
      { status: 404 }
    );
  }

  // Don't actually delete, just mark as cancelled
  subscription.status = 'cancelled';
  subscription.cancellationDate = new Date();
  subscription.autoRenew = false;
  subscription.updatedAt = new Date();

  subscriptions.set(userId, subscription);

  return NextResponse.json({
    success: true,
    message: 'Subscription cancelled. Credits remain until renewal date.',
    data: subscription,
  });
}
